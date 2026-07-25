"""CIFAR-10 active block-acquisition experiment (domain 2 of the PVAS paper).

Protocol (retrospective, matched budget): every method starts from the SAME
random initial mask (frac0 of 4x4 blocks observed) and ends with the SAME
total number of observed blocks (frac0 + budget). Policies differ only in
WHICH blocks they acquire mid-trajectory (and 'static' acquires them all
up-front at random).  All methods share the same pretrained DDPM prior.

Run:   python -m poc_natural.run_cifar --n_images 64 --budget_blocks 12
Writes outputs/poc_natural/cifar_<tag>/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poc_natural.pvas_core import (  # noqa: E402
    BlockMaskState, DenoiserStats, EpsDenoiser, PVASConfig, PVASResult,
    acquire_blocks, init_block_state, make_abar, pv_estimate, run_pvas,
)


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PSNR on [0,1]-scaled images, per batch element."""
    a = (a.clamp(-1, 1) + 1) / 2
    b = (b.clamp(-1, 1) + 1) / 2
    mse = (a - b).pow(2).flatten(1).mean(dim=1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-12))


def ssim_batch(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    from skimage.metrics import structural_similarity
    a = ((a.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    b = ((b.clamp(-1, 1) + 1) / 2).permute(0, 2, 3, 1).cpu().numpy()
    return np.array([
        structural_similarity(ai, bi, channel_axis=2, data_range=1.0)
        for ai, bi in zip(a, b)
    ])


@torch.no_grad()
def run_ensemble_active(unet, abar, state: BlockMaskState, x0_full: torch.Tensor,
                        budget_blocks: int, cfg: PVASConfig, K: int,
                        select: str = "var") -> tuple[PVASResult, int]:
    """ADS/AdaSense-style expensive reference: K parallel posterior trajectories;
    acquisition scores = cross-particle variance of x0hat (select="var") or
    random (select="random", isolating the pure particle-averaging benefit);
    recon = particle mean. NFE = K * num_steps (no probes needed).
    """
    B = x0_full.shape[0]
    xrep = x0_full.repeat_interleave(K, dim=0)
    st = BlockMaskState(state.mask.repeat_interleave(K, dim=0),
                        state.y.repeat_interleave(K, dim=0), state.sigma_y, state.block,
                        y_lr=None if state.y_lr is None
                        else state.y_lr.repeat_interleave(K, dim=0))
    stats = DenoiserStats()
    den = EpsDenoiser(unet, abar, stats)

    pol_map = {"var": "ensemble", "random": "random", "duo": "duo"}
    ens_cfg = PVASConfig(**{**cfg.__dict__, "policy": pol_map[select],
                            "use_pv_gate": False, "pv_every": 10 ** 9})

    # Policy hook: cross-particle variance of x0hat scores the blocks.
    from poc_natural import pvas_core as core

    orig_score = core._score_blocks

    def patched(policy, den_, xt, t, x0hat, x0f, st_, pv, cfg_, gen, ens):
        if policy == "duo":
            # PVAS-duo: total-variance score from K=2 trajectories ---
            # within-mode Tweedie PV (particle 1) + cross-trajectory
            # disagreement (unbiased 2-sample variance). Law of total
            # variance: Var_total = E[within-mode] + Var[mode means]; the
            # disagreement term restores the cross-mode component that a
            # single trajectory cannot see.
            block = st_.block
            Bk = x0hat.shape[0] // K
            x0v = x0hat.view(Bk, K, *x0hat.shape[1:])
            disagree = 0.5 * (x0v[:, 0] - x0v[:, 1]).pow(2).mean(1, keepdim=True)
            xt_first = xt.view(Bk, K, *xt.shape[1:])[:, 0]
            pv1 = pv_estimate(den_, xt_first, t, n_probes=cfg_.pv_n_probes,
                              smooth_sigma=cfg_.pv_smooth_sigma)
            total = pv1 + disagree
            total = total.repeat_interleave(K, dim=0)
            return torch.nn.functional.avg_pool2d(total, block) * (block ** 2)
        if policy != "ensemble":
            return orig_score(policy, den_, xt, t, x0hat, x0f, st_, pv, cfg_, gen, ens)
        block = st_.block
        Bk = x0hat.shape[0] // K
        var = x0hat.view(Bk, K, *x0hat.shape[1:]).var(dim=1).mean(dim=1, keepdim=True)
        var = var.repeat_interleave(K, dim=0)
        return torch.nn.functional.avg_pool2d(var, block) * (block ** 2)

    core._score_blocks = patched
    try:
        res = run_pvas(den, st, xrep, budget_blocks, ens_cfg)
    finally:
        core._score_blocks = orig_score
    xmean = res.x0.view(B, K, *res.x0.shape[1:]).mean(dim=1)
    out = PVASResult(x0=xmean, state=res.state, nfe=stats.nfe)
    return out, stats.nfe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--mode", choices=["inpaint", "sr"], default="inpaint",
                   help="'sr': low-res image always observed (adaptive "
                        "super-resolution); budget buys full-res blocks")
    p.add_argument("--frac0", type=float, default=0.10)
    p.add_argument("--budget_blocks", type=int, default=12)
    p.add_argument("--sigma_y", type=float, default=0.05)
    p.add_argument("--active_rounds", type=int, default=2)
    p.add_argument("--pv_every", type=int, default=5)
    p.add_argument("--ensemble_k", type=int, default=8)
    p.add_argument("--pv_probes", type=int, default=4,
                   help="Hutchinson probes for the pv policy (2*probes NFE per refresh)")
    p.add_argument("--policies", nargs="+",
                   default=["static", "random", "raster", "pv", "oracle",
                            "ensemble", "ens_random", "duo"])
    p.add_argument("--pv_gate", action="store_true",
                   help="PV-gated DC for the 'pv' policy (ablation); default: "
                        "scalar gain for all policies so the main table isolates "
                        "the acquisition policy at near-matched NFE.")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="v1")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device).eval()
    sched = DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
    abar = make_abar(sched).to(device)

    cache = Path(__file__).resolve().parents[1] / "data" / "cifar10" / "test_images_512.pt"
    all_imgs = torch.load(cache)  # first 512 CIFAR-10 test images in [-1,1]
    idx = torch.randperm(len(all_imgs),
                         generator=torch.Generator().manual_seed(123))[: args.n_images]
    images = all_imgs[idx]

    gh = gw = 32 // args.block
    n_total = gh * gw
    n0 = max(1, int(round(args.frac0 * n_total)))
    out_dir = (Path(__file__).resolve().parents[1] / "outputs" / "poc_natural"
               / f"cifar_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"config": vars(args), "per_image": {}, "summary": {}, "nfe": {}}
    t_start = time.time()

    for pol in args.policies:
        psnrs, ssims = [], []
        nfe_total = 0
        for b0 in range(0, args.n_images, args.batch):
            x0 = images[b0: b0 + args.batch].to(device)
            gen = torch.Generator().manual_seed(args.seed + 1000 + b0)
            state0 = init_block_state(x0, args.frac0, args.block, args.sigma_y, gen,
                                      with_lr=(args.mode == "sr"))

            cfg = PVASConfig(num_steps=args.num_steps, pv_eps_probe=5e-2,
                             pv_n_probes=args.pv_probes, pv_every=args.pv_every,
                             use_pv_gate=args.pv_gate and pol == "pv" and args.sigma_y > 0,
                             active_rounds=args.active_rounds,
                             policy=pol if pol not in ("static",) else "random",
                             seed=args.seed + b0)

            if pol == "static":
                # all budget up-front at random; no mid-trajectory acquisition
                gen2 = torch.Generator().manual_seed(args.seed + 2000 + b0)
                scores = torch.rand(x0.shape[0], 1, gh, gw,
                                    generator=gen2).to(device)
                state0 = acquire_blocks(state0, x0, scores, args.budget_blocks, gen2)
                stats = DenoiserStats()
                den = EpsDenoiser(unet, abar, stats)
                cfg_static = PVASConfig(**{**cfg.__dict__, "active_rounds": 0,
                                           "policy": "random"})
                res = run_pvas(den, state0, x0, 0, cfg_static)
                nfe_total += stats.nfe
            elif pol in ("ensemble", "ens_random", "duo"):
                sel = {"ensemble": "var", "ens_random": "random", "duo": "duo"}[pol]
                kk = 2 if pol == "duo" else args.ensemble_k
                res, nfe = run_ensemble_active(
                    unet, abar, state0, x0, args.budget_blocks, cfg, kk, select=sel)
                nfe_total += nfe
            else:
                stats = DenoiserStats()
                den = EpsDenoiser(unet, abar, stats)
                res = run_pvas(den, state0, x0, args.budget_blocks, cfg)
                nfe_total += stats.nfe

            psnrs.extend(psnr(res.x0, x0).cpu().tolist())
            ssims.extend(ssim_batch(res.x0, x0).tolist())
            print(f"[{pol}] {b0 + x0.shape[0]}/{args.n_images} "
                  f"psnr_mean={np.mean(psnrs):.3f} ({time.time() - t_start:.0f}s)",
                  flush=True)

        results["per_image"][pol] = {"psnr": psnrs, "ssim": ssims}
        results["nfe"][pol] = nfe_total / args.n_images
        results["summary"][pol] = {
            "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
            "nfe_per_image": nfe_total / args.n_images,
        }
        print(f"== {pol}: PSNR {np.mean(psnrs):.3f}±{np.std(psnrs):.3f} "
              f"SSIM {np.mean(ssims):.4f} NFE/img {nfe_total / args.n_images:.0f}",
              flush=True)

    # paired deltas vs matched-budget baselines
    if "pv" in results["per_image"]:
        from scipy.stats import wilcoxon
        pvp = np.array(results["per_image"]["pv"]["psnr"])
        for ref in ["static", "random", "raster", "ensemble", "oracle"]:
            if ref in results["per_image"]:
                d = pvp - np.array(results["per_image"][ref]["psnr"])
                try:
                    w = wilcoxon(d, alternative="greater").pvalue
                except Exception:
                    w = float("nan")
                results["summary"][f"delta_pv_vs_{ref}"] = {
                    "mean": float(d.mean()), "std": float(d.std()),
                    "wilcoxon_p_greater": float(w)}
                print(f"Δ(pv - {ref}) = {d.mean():+.3f} ± {d.std():.3f} dB "
                      f"(Wilcoxon p={w:.2e})", flush=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_dir / 'results.json'}  total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
