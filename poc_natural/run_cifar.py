"""CIFAR-10 development experiment for SketchSense block acquisition.

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
    acquire_blocks, init_block_state, make_abar, pv_estimate,
    random_block_scores, run_pvas,
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
def lpips_batch(metric, a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """LPIPS-Alex values for images already scaled to [-1, 1]."""
    return metric(a.clamp(-1, 1), b.clamp(-1, 1)).flatten().cpu().tolist()


def reset_cuda_peak_memory(device: torch.device) -> None:
    """Start a synchronized CUDA peak-memory measurement."""
    synchronize_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def synchronize_device(device: torch.device) -> None:
    """Synchronize CUDA for honest wall-clock measurements."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_peak_memory_mb(device: torch.device) -> float | None:
    """Return synchronized peak allocated CUDA memory, or None off CUDA."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def paired_bootstrap_mean_ci(
    differences,
    *,
    seed: int = 0,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Deterministic percentile interval for a paired mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a nonempty one-dimensional array")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, values.size, size=(int(resamples), values.size)
    )
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


@torch.no_grad()
def run_canonical_replay(
    unet,
    abar: torch.Tensor,
    final_state: BlockMaskState,
    x0_full: torch.Tensor,
    cfg: PVASConfig,
) -> tuple[PVASResult, int]:
    """Reconstruct a selected final mask with one fixed passive solver.

    Active and ensemble policies can improve their system-level reconstruction
    for reasons other than mask quality. This replay exposes every final mask
    up front and runs the same single-trajectory, scalar-DC sampler so the
    primary selection comparison is not confounded by posterior averaging or
    acquisition timing.
    """
    stats = DenoiserStats()
    denoiser = EpsDenoiser(unet, abar, stats)
    replay_cfg = PVASConfig(**{
        **cfg.__dict__,
        "active_rounds": 0,
        "policy": "random",
        "use_pv_gate": False,
    })
    result = run_pvas(
        denoiser,
        final_state,
        x0_full,
        budget_blocks=0,
        cfg=replay_cfg,
    )
    return result, stats.nfe


def observed_block_ids(state: BlockMaskState) -> list[list[int]]:
    """Serialize the selected atom set for every image in a batch."""
    observed = (
        torch.nn.functional.avg_pool2d(state.mask, state.block)
        .flatten(1) > 0.5
    )
    return [
        torch.nonzero(row, as_tuple=False).flatten().cpu().tolist()
        for row in observed
    ]


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
                        else state.y_lr.repeat_interleave(K, dim=0),
                        y_full=None if state.y_full is None
                        else state.y_full.repeat_interleave(K, dim=0))
    stats = DenoiserStats()
    den = EpsDenoiser(unet, abar, stats)

    pol_map = {"var": "ensemble", "random": "random", "duo": "duo"}
    ens_cfg = PVASConfig(**{**cfg.__dict__, "policy": pol_map[select],
                            "use_pv_gate": False, "pv_every": 10 ** 9})

    # Policy hook: cross-particle variance of x0hat scores the blocks.
    from poc_natural import pvas_core as core

    orig_score = core._score_blocks
    def patched(
        policy, den_, xt, t, x0hat, x0f, st_, pv, cfg_, gen, ens, k=None
    ):
        if policy == "duo":
            # Exploratory two-trajectory score: local Tweedie uncertainty from
            # particle 1 plus between-trajectory disagreement. The latter is a
            # two-sample variance of solver outputs, but the current DDNM-style
            # trajectories are not proven draws from the required conditional
            # mode distribution. Do not call this an exact law-of-total-
            # variance posterior decomposition without further validation.
            block = st_.block
            Bk = x0hat.shape[0] // K
            x0v = x0hat.view(Bk, K, *x0hat.shape[1:])
            disagree = 0.5 * (x0v[:, 0] - x0v[:, 1]).pow(2).mean(1, keepdim=True)
            xt_first = xt.view(Bk, K, *xt.shape[1:])[:, 0]
            duo_probe_gen = torch.Generator(device="cpu").manual_seed(
                int(
                    (int(cfg_.seed) * 1_000_003 + int(t) * 104_729 + 31)
                    % (2**63 - 1)
                )
            )
            pv1 = pv_estimate(den_, xt_first, t, n_probes=cfg_.pv_n_probes,
                              smooth_sigma=cfg_.pv_smooth_sigma,
                              generator=duo_probe_gen, clamp_final=False)
            total = pv1 + disagree
            total = (
                torch.nn.functional.avg_pool2d(total, block) * (block ** 2)
            ).clamp_min(0.0)
            return total.repeat_interleave(K, dim=0)
        if policy == "random":
            # All particles acquire the same atoms. Independent per-particle
            # masks would leak a much larger union of measurements into the
            # particle mean and invalidate this control.
            Bk = x0hat.shape[0] // K
            gh = x0hat.shape[-2] // st_.block
            gw = x0hat.shape[-1] // st_.block
            base = random_block_scores(Bk, gh, gw, cfg_.seed, x0hat.device)
            return base.repeat_interleave(K, dim=0)
        if policy != "ensemble":
            return orig_score(
                policy, den_, xt, t, x0hat, x0f, st_, pv, cfg_, gen, ens, k=k
            )
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
    mask_bk = res.state.mask.view(B, K, *res.state.mask.shape[1:])
    y_bk = res.state.y.view(B, K, *res.state.y.shape[1:])
    if not torch.equal(mask_bk, mask_bk[:, :1].expand_as(mask_bk)):
        raise AssertionError("ensemble particles ended with different masks")
    if not torch.equal(y_bk, y_bk[:, :1].expand_as(y_bk)):
        raise AssertionError("ensemble particles ended with different measurements")
    collapsed_state = BlockMaskState(
        mask=mask_bk[:, 0].clone(),
        y=y_bk[:, 0].clone(),
        sigma_y=res.state.sigma_y,
        block=res.state.block,
        y_lr=None if res.state.y_lr is None else (
            res.state.y_lr.view(B, K, *res.state.y_lr.shape[1:])[:, 0].clone()
        ),
        y_full=None if res.state.y_full is None else (
            res.state.y_full.view(B, K, *res.state.y_full.shape[1:])[:, 0].clone()
        ),
    )
    out = PVASResult(x0=xmean, state=collapsed_state, nfe=stats.nfe)
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
    p.add_argument("--sketch_rank", type=int, default=4,
                   help="Real-embedded probes for operator_diag/sketch policies")
    p.add_argument("--sketch_fd_step", type=float, default=5e-2)
    p.add_argument("--sketch_psd_rtol", type=float, default=1e-6)
    p.add_argument("--sketch_noise_variance_floor", type=float, default=1e-4,
                   help="Design regularizer for noiseless measurements")
    p.add_argument("--policies", nargs="+",
                   default=["static", "random", "raster", "pv", "oracle",
                            "ensemble", "ens_random", "duo"])
    p.add_argument("--pv_gate", action="store_true",
                   help="PV-gated DC for the 'pv' policy (ablation); default: "
                        "scalar gain for all policies so the main table isolates "
                        "the acquisition policy. Actual unequal NFE is reported.")
    p.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute LPIPS-Alex for system and canonical reconstructions.",
    )
    p.add_argument(
        "--skip_canonical_replay",
        action="store_true",
        help="Pilot-only speed option. By default every selected final mask is "
             "reconstructed by the same passive single-trajectory solver.",
    )
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
    lpips_metric = None
    if args.lpips:
        import lpips as lpips_library
        lpips_metric = lpips_library.LPIPS(net="alex").to(device).eval()

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

    results = {
        "status": "CORRECTED_RUN_UNREVIEWED",
        "config": vars(args),
        "protocol": {
            "primary_selection_metric": (
                "system reconstruction"
                if args.skip_canonical_replay
                else "canonical single-trajectory replay of the final mask"
            ),
            "system_and_selection_metrics_separated": True,
        },
        "per_image": {},
        "summary": {},
        "nfe": {},
    }
    t_start = time.time()

    for pol in args.policies:
        system_psnrs, system_ssims = [], []
        canonical_psnrs, canonical_ssims = [], []
        system_lpips, canonical_lpips = [], []
        final_atom_ids: list[list[int]] = []
        nfe_total = 0
        canonical_nfe_total = 0
        system_wall_time = 0.0
        canonical_wall_time = 0.0
        system_peak_memory_mb = None
        canonical_peak_memory_mb = None
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
                             sketch_rank=args.sketch_rank,
                             sketch_fd_step=args.sketch_fd_step,
                             sketch_psd_rtol=args.sketch_psd_rtol,
                             sketch_noise_variance_floor=(
                                 args.sketch_noise_variance_floor
                             ),
                             seed=args.seed + b0)

            reset_cuda_peak_memory(device)
            system_started_at = time.time()
            if pol == "static":
                # Reveal the same fixed random ranking used by the adaptive
                # random policy, but all up-front: a timing-only control.
                gen2 = torch.Generator().manual_seed(cfg.seed + 2_000_003)
                scores = random_block_scores(
                    x0.shape[0], gh, gw, cfg.seed, device
                )
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

            batch_system_peak = cuda_peak_memory_mb(device)
            system_wall_time += float(time.time() - system_started_at)
            if batch_system_peak is not None:
                system_peak_memory_mb = max(
                    system_peak_memory_mb or 0.0,
                    batch_system_peak,
                )
            system_psnrs.extend(psnr(res.x0, x0).cpu().tolist())
            system_ssims.extend(ssim_batch(res.x0, x0).tolist())
            if lpips_metric is not None:
                system_lpips.extend(lpips_batch(lpips_metric, res.x0, x0))
            final_atom_ids.extend(observed_block_ids(res.state))

            if args.skip_canonical_replay:
                selection_result = res
            elif pol == "static":
                # Static already exposes its final mask from the first step and
                # uses the canonical passive solver.
                selection_result = res
            else:
                reset_cuda_peak_memory(device)
                canonical_started_at = time.time()
                selection_result, replay_nfe = run_canonical_replay(
                    unet, abar, res.state, x0, cfg
                )
                canonical_nfe_total += replay_nfe
                batch_canonical_peak = cuda_peak_memory_mb(device)
                canonical_wall_time += float(
                    time.time() - canonical_started_at
                )
                if batch_canonical_peak is not None:
                    canonical_peak_memory_mb = max(
                        canonical_peak_memory_mb or 0.0,
                        batch_canonical_peak,
                    )
            canonical_psnrs.extend(
                psnr(selection_result.x0, x0).cpu().tolist()
            )
            canonical_ssims.extend(
                ssim_batch(selection_result.x0, x0).tolist()
            )
            if lpips_metric is not None:
                canonical_lpips.extend(
                    lpips_batch(lpips_metric, selection_result.x0, x0)
                )
            print(f"[{pol}] {b0 + x0.shape[0]}/{args.n_images} "
                  f"selection_psnr={np.mean(canonical_psnrs):.3f} "
                  f"system_psnr={np.mean(system_psnrs):.3f} "
                  f"({time.time() - t_start:.0f}s)",
                  flush=True)

        results["per_image"][pol] = {
            # Backward-compatible keys are always the primary selection
            # comparison defined in results["protocol"].
            "psnr": canonical_psnrs,
            "ssim": canonical_ssims,
            "system_psnr": system_psnrs,
            "system_ssim": system_ssims,
            "final_atom_ids": final_atom_ids,
        }
        if lpips_metric is not None:
            results["per_image"][pol].update({
                "lpips": canonical_lpips,
                "system_lpips": system_lpips,
            })
        results["nfe"][pol] = {
            "system_nfe_per_image": nfe_total / args.n_images,
            "evaluation_only_canonical_replay_nfe_per_image": (
                canonical_nfe_total / args.n_images
            ),
        }
        results["summary"][pol] = {
            "psnr_mean": float(np.mean(canonical_psnrs)),
            "psnr_std": float(np.std(canonical_psnrs)),
            "ssim_mean": float(np.mean(canonical_ssims)),
            "ssim_std": float(np.std(canonical_ssims)),
            "system_psnr_mean": float(np.mean(system_psnrs)),
            "system_psnr_std": float(np.std(system_psnrs)),
            "system_ssim_mean": float(np.mean(system_ssims)),
            "system_ssim_std": float(np.std(system_ssims)),
            "system_nfe_per_image": nfe_total / args.n_images,
            "evaluation_only_canonical_replay_nfe_per_image": (
                canonical_nfe_total / args.n_images
            ),
            "system_wall_time_seconds": system_wall_time,
            "system_peak_cuda_memory_mb": system_peak_memory_mb,
            "evaluation_only_canonical_replay_wall_time_seconds": (
                canonical_wall_time
            ),
            "evaluation_only_canonical_replay_peak_cuda_memory_mb": (
                canonical_peak_memory_mb
            ),
        }
        if lpips_metric is not None:
            results["summary"][pol].update({
                "lpips_mean": float(np.mean(canonical_lpips)),
                "lpips_std": float(np.std(canonical_lpips)),
                "system_lpips_mean": float(np.mean(system_lpips)),
                "system_lpips_std": float(np.std(system_lpips)),
            })
        print(f"== {pol}: selection PSNR "
              f"{np.mean(canonical_psnrs):.3f}±{np.std(canonical_psnrs):.3f} "
              f"system PSNR {np.mean(system_psnrs):.3f} "
              f"NFE/img {nfe_total / args.n_images:.0f}",
              flush=True)

    # Paired per-image deltas. Unequal NFE remains explicit in the summary.
    for candidate in [
        "operator_diag", "sketch_marginal", "sketch_logdet", "pv"
    ]:
        if candidate not in results["per_image"]:
            continue
        from scipy.stats import wilcoxon
        candidate_psnr = np.array(results["per_image"][candidate]["psnr"])
        references = [
            "static", "random", "raster", "operator_diag",
            "sketch_marginal", "ensemble", "oracle",
        ]
        for ref in references:
            if ref == candidate or ref not in results["per_image"]:
                continue
            d = candidate_psnr - np.array(results["per_image"][ref]["psnr"])
            try:
                w = wilcoxon(d, alternative="greater").pvalue
            except Exception:
                w = float("nan")
            ci_low, ci_high = paired_bootstrap_mean_ci(
                d,
                seed=args.seed + 91,
            )
            results["summary"][f"delta_{candidate}_vs_{ref}"] = {
                "mean": float(d.mean()), "std": float(d.std()),
                "paired_bootstrap_95ci": [ci_low, ci_high],
                "wilcoxon_p_greater": float(w)}
            print(
                f"Δ({candidate} - {ref}) = {d.mean():+.3f} "
                f"± {d.std():.3f} dB (Wilcoxon p={w:.2e})",
                flush=True,
            )

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_dir / 'results.json'}  total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
