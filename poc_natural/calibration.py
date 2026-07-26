"""Estimator-quality experiment: is single-probe Tweedie PV a good uncertainty signal?

For each test image and diffusion time t we compare, per 4x4 block:
  (a) ours: single-trajectory Tweedie PV (1 Hutchinson probe, 2 NFE),
  (b) ours-k: same with k probes (2k NFE),
  (c) ensemble: variance of x0hat across K independent posterior-sampling
      trajectories continued from independent noise (ADS/AdaSense-style,
      K * (steps so far) NFE),
  (d) oracle: true squared error of the current x0hat.

Reported: Spearman rank correlation of (a,b,c) against (d) per image
(selection quality is rank-based: what matters is ranking blocks), and
top-k overlap with the oracle's top-k blocks, plus NFE cost of each signal.

Run: python -m poc_natural.calibration --n_images 32
Writes outputs/poc_natural/calibration/results.json
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
    BlockMaskState, DenoiserStats, EpsDenoiser, PVASConfig,
    init_block_state, make_abar, pv_estimate, run_pvas, ddim_timesteps, _dc_update,
)


@torch.no_grad()
def partial_trajectory(den, state, x0_full, cfg, stop_frac, gen):
    """Run DDIM+DC up to stop_frac of the schedule; return (xt, t, i, ts)."""
    device = x0_full.device
    abar = den.abar
    ts = ddim_timesteps(len(abar), cfg.num_steps)
    xt = torch.randn(x0_full.shape, generator=gen).to(device)
    stop_i = int(stop_frac * (cfg.num_steps - 1))
    for i, t in enumerate(ts.tolist()):
        a_t = abar[t].to(device)
        x0hat, eps_theta = den.predict(xt, t)
        x0hat = _dc_update(x0hat, state, torch.ones(1, 1, 1, 1, device=device))
        if i >= stop_i:
            return xt, t, x0hat
        t_next = ts[i + 1].item()
        a_n = abar[t_next].to(device)
        xt = torch.sqrt(a_n) * x0hat + torch.sqrt(1 - a_n) * eps_theta
    return xt, ts[-1].item(), x0hat


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=32)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--mode", choices=["inpaint", "sr"], default="inpaint")
    p.add_argument("--frac0", type=float, default=0.10)
    p.add_argument("--sigma_y", type=float, default=0.05)
    p.add_argument("--stop_frac", type=float, default=0.5)
    p.add_argument("--probes", type=int, nargs="+", default=[1, 4])
    p.add_argument("--ensemble_ks", type=int, nargs="+", default=[16])
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to(device).eval()
    abar = make_abar(DDPMScheduler.from_pretrained("google/ddpm-cifar10-32")).to(device)

    cache = Path(__file__).resolve().parents[1] / "data" / "cifar10" / "test_images_512.pt"
    all_imgs = torch.load(cache)  # first 512 CIFAR-10 test images in [-1,1]
    idx = torch.randperm(len(all_imgs),
                         generator=torch.Generator().manual_seed(123))[: args.n_images]
    images = all_imgs[idx]

    cfg = PVASConfig(num_steps=args.num_steps)
    block = args.block
    rows: dict = {}
    top_overlap: dict = {}
    nfe_cost = {}
    t_start = time.time()

    for b0 in range(0, args.n_images, args.batch):
        x0 = images[b0: b0 + args.batch].to(device)
        B = x0.shape[0]
        gen = torch.Generator().manual_seed(args.seed + 1000 + b0)
        state = init_block_state(x0, args.frac0, block, args.sigma_y, gen,
                                 with_lr=(args.mode == "sr"))

        # shared partial trajectory to the probe point
        stats = DenoiserStats()
        den = EpsDenoiser(unet, abar, stats)
        xt, t, x0hat = partial_trajectory(den, state, x0, cfg, args.stop_frac, gen)
        base_nfe = stats.nfe / B

        # (d) oracle block error of current x0hat
        err = (x0hat - x0).pow(2).mean(1, keepdim=True)
        err_b = torch.nn.functional.avg_pool2d(err, block).flatten(1).cpu().numpy()

        signals = {}
        # (a,b) PV at each probe count
        for npb in args.probes:
            sp = DenoiserStats(); dp = EpsDenoiser(unet, abar, sp)
            probe_gen = torch.Generator(device="cpu").manual_seed(
                args.seed + 3_000_001 + b0
            )
            sig = pv_estimate(dp, xt, t, n_probes=npb,
                              pool_block=block,
                              generator=probe_gen).flatten(1).cpu().numpy()
            signals[f"pv{npb}"] = sig
            nfe_cost[f"pv{npb}"] = sp.nfe / B

        # (c) ensembles: K independent trajectories to the same stop point
        for K in args.ensemble_ks:
            se = DenoiserStats(); de = EpsDenoiser(unet, abar, se)
            x0r = x0.repeat_interleave(K, 0)
            str_ = BlockMaskState(state.mask.repeat_interleave(K, 0),
                                  state.y.repeat_interleave(K, 0), state.sigma_y, block,
                                  y_lr=None if state.y_lr is None
                                  else state.y_lr.repeat_interleave(K, 0),
                                  y_full=None if state.y_full is None
                                  else state.y_full.repeat_interleave(K, 0))
            genk = torch.Generator().manual_seed(args.seed + 5000 + b0 + K)
            _, _, x0hat_k = partial_trajectory(de, str_, x0r, cfg, args.stop_frac, genk)
            var_k = x0hat_k.view(B, K, *x0hat_k.shape[1:]).var(1).mean(1, keepdim=True)
            signals[f"ens{K}"] = (torch.nn.functional.avg_pool2d(var_k, block)
                                  .flatten(1).cpu().numpy())
            nfe_cost[f"ens{K}"] = se.nfe / B

        # unobserved blocks only (selection candidates)
        pooled_mask = (torch.nn.functional.avg_pool2d(state.mask, block)
                       .flatten(1).cpu().numpy() > 0.5)
        for bi in range(B):
            unobs = ~pooled_mask[bi]
            if unobs.sum() < args.topk + 2:
                continue
            e = err_b[bi][unobs]
            ok = np.argsort(-e)[: args.topk]
            for name, sig in signals.items():
                s = sig[bi][unobs]
                rows.setdefault(name, []).append(spearman(s, e))
                sk = np.argsort(-s)[: args.topk]
                top_overlap.setdefault(name, []).append(len(set(ok) & set(sk)) / args.topk)
        print(f"{b0 + B}/{args.n_images} done ({time.time() - t_start:.0f}s)", flush=True)

    out = {"config": vars(args), "base_nfe_to_probe_point": base_nfe,
           "signals": {}}
    for name in rows:
        out["signals"][name] = {
            "spearman_vs_oracle_mean": float(np.mean(rows[name])),
            "spearman_vs_oracle_std": float(np.std(rows[name])),
            "topk_overlap_mean": float(np.mean(top_overlap[name])),
            "extra_nfe_per_image": nfe_cost[name],
        }
        print(f"{name}: spearman {np.mean(rows[name]):.3f}±{np.std(rows[name]):.3f} "
              f"top-{args.topk} overlap {np.mean(top_overlap[name]):.3f} "
              f"extra NFE {nfe_cost[name]:.0f}")
    out_dir = (Path(__file__).resolve().parents[1] / "outputs" / "poc_natural"
               / f"calibration_{args.mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_dir / "results.json")


if __name__ == "__main__":
    main()
