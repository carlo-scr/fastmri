"""CelebA-HQ 256x256 QUANTITATIVE guided-SR experiment (paper tab:celeba).

Same protocol as run_cifar --mode sr, at 8x scale: the 8x box-downsampled
32x32 preview is always observed; the budget buys 8x8-pixel full-resolution
blocks (grid 32x32 = 1024 atoms). Policies at matched budget with NFE
accounting. Prior: google/ddpm-celebahq-256 (unconditional).

Run: python -m poc_natural.quantitative_celeba --n_images 64
Writes outputs/poc_natural/celeba_quant/results.json
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
    DenoiserStats, EpsDenoiser, PVASConfig, acquire_blocks, init_block_state,
    make_abar, run_pvas,
)
from poc_natural.run_cifar import psnr, ssim_batch, run_ensemble_active  # noqa: E402
from poc_natural.qualitative_celeba import load_faces  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=64)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--budget_blocks", type=int, default=96)
    p.add_argument("--active_rounds", type=int, default=3)
    p.add_argument("--pv_probes", type=int, default=4)
    p.add_argument("--ensemble_k", type=int, default=4)
    p.add_argument("--policies", nargs="+",
                   default=["static", "random", "pv", "oracle", "duo", "ensemble"])
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="quant")
    args = p.parse_args()

    device = torch.device(args.device)
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device).eval()
    abar = make_abar(DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")).to(device)

    faces_cache = (Path(__file__).resolve().parents[1] / "data"
                   / f"celeba_hq_{args.n_images}.pt")
    if faces_cache.exists():
        images = torch.load(faces_cache)
    else:
        images = load_faces(args.n_images)
        torch.save(images, faces_cache)
    print(f"faces: {images.shape}", flush=True)

    gh = 256 // args.block
    out_dir = (Path(__file__).resolve().parents[1] / "outputs" / "poc_natural"
               / f"celeba_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"config": vars(args), "per_image": {}, "summary": {}, "nfe": {}}
    t_start = time.time()

    for pol in args.policies:
        psnrs, ssims = [], []
        nfe_total = 0
        for b0 in range(0, args.n_images, args.batch):
            x0 = images[b0: b0 + args.batch].to(device)
            gen = torch.Generator().manual_seed(args.seed + 1000 + b0)
            state0 = init_block_state(x0, 0.0, args.block, 0.0, gen, with_lr=True)
            cfg = PVASConfig(num_steps=args.num_steps, pv_n_probes=args.pv_probes,
                             use_pv_gate=False, active_rounds=args.active_rounds,
                             active_window=(0.25, 0.65),
                             policy=pol if pol not in ("static",) else "random",
                             seed=args.seed + b0)
            if pol == "static":
                g2 = torch.Generator().manual_seed(args.seed + 2000 + b0)
                sc = torch.rand(x0.shape[0], 1, gh, gh, generator=g2).to(device)
                state0 = acquire_blocks(state0, x0, sc, args.budget_blocks, g2)
                stats = DenoiserStats()
                res = run_pvas(EpsDenoiser(unet, abar, stats), state0, x0, 0,
                               PVASConfig(**{**cfg.__dict__, "active_rounds": 0,
                                             "policy": "random"}))
                nfe_total += stats.nfe
            elif pol in ("ensemble", "duo"):
                kk = 2 if pol == "duo" else args.ensemble_k
                sel = "duo" if pol == "duo" else "var"
                res, nfe = run_ensemble_active(unet, abar, state0, x0,
                                               args.budget_blocks, cfg, kk, select=sel)
                nfe_total += nfe
            else:
                stats = DenoiserStats()
                res = run_pvas(EpsDenoiser(unet, abar, stats), state0, x0,
                               args.budget_blocks, cfg)
                nfe_total += stats.nfe
            psnrs.extend(psnr(res.x0, x0).cpu().tolist())
            ssims.extend(ssim_batch(res.x0, x0).tolist())
            print(f"[{pol}] {b0 + x0.shape[0]}/{args.n_images} "
                  f"psnr={np.mean(psnrs):.3f} ({time.time() - t_start:.0f}s)",
                  flush=True)
        results["per_image"][pol] = {"psnr": psnrs, "ssim": ssims}
        results["nfe"][pol] = nfe_total / args.n_images
        results["summary"][pol] = {
            "psnr_mean": float(np.mean(psnrs)), "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)), "ssim_std": float(np.std(ssims)),
            "nfe_per_image": nfe_total / args.n_images}
        print(f"== {pol}: PSNR {np.mean(psnrs):.3f}±{np.std(psnrs):.3f} "
              f"SSIM {np.mean(ssims):.4f} NFE {nfe_total / args.n_images:.0f}",
              flush=True)

    from scipy.stats import wilcoxon
    if "pv" in results["per_image"]:
        pvp = np.array(results["per_image"]["pv"]["psnr"])
        for ref in ["static", "random", "duo", "ensemble", "oracle"]:
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
                      f"(p={w:.2e})", flush=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out_dir / "results.json", f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
