"""Smoke test for posterior-variance-gated FA-KGD on fastMRI brain MC.

Compares (on the same noisy multi-coil k-space, same slice):

  (1) ΠGDM-MC                           [baseline]
  (2) FA-KGD-MC                         [previous method]
  (3) FA-KGD-MC + PV gate (A only)
  (4) FA-KGD-MC + PV gate + sens-refine (A+B)
  (5) FA-KGD-MC + PV gate + sens-refine + active lines (A+B+D)

Reports PSNR vs RSS_GT and vs SENSE-oracle for all five, plus the
deltas relative to ΠGDM-MC.

Defaults: 1 vol, 2 slices, T=20, R=4, brain. ~1 min/method/slice on CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.mri_forward import fft2c, ifft2c, create_mask, build_radius_grid
from src.samplers.schedules import edm_sigma_schedule
from src.samplers.sense import (
    estimate_sens_maps_lowres, estimate_noise_per_coil, MultiCoilSENSE,
    sense_combine,
)
from src.samplers.multicoil import run_pigdm_mc, run_fakgd_mc
from src.samplers.multicoil_pv import run_fakgd_mc_pv
from src.models.edm_loader import load_edm_model, EDMDenoiser


def _center_crop(x, h, w):
    H, W = x.shape[-2], x.shape[-1]
    sh = (H - h) // 2; sw = (W - w) // 2
    return x[..., sh:sh+h, sw:sw+w]


def _psnr(gt_mag, recon_mag):
    H = min(gt_mag.shape[-2], 320); W = min(gt_mag.shape[-1], 320)
    gt = _center_crop(gt_mag, H, W); rc = _center_crop(recon_mag, H, W)
    dr = gt.max(); mse = ((gt - rc) ** 2).mean()
    return 100.0 if mse < 1e-12 else (10 * torch.log10(dr ** 2 / mse)).item()


def load_multicoil(h5_path, slice_idx, target_shape):
    with h5py.File(h5_path, "r") as f:
        ks = torch.from_numpy(f["kspace"][slice_idx])
    coil_imgs_full = ifft2c(ks)
    coil_imgs = _center_crop(coil_imgs_full, target_shape[0], target_shape[1])
    rss = torch.sqrt((coil_imgs.abs() ** 2).sum(dim=0))
    scale = float(rss.max().clamp(min=1e-30))
    coil_imgs = coil_imgs / scale
    rss = rss / scale
    return fft2c(coil_imgs), rss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/brain_val")
    p.add_argument("--checkpoint_dir", default="checkpoints/edm/supervised_R=1")
    p.add_argument("--num_volumes", type=int, default=1)
    p.add_argument("--num_slices", type=int, default=2)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--target_h", type=int, default=384)
    p.add_argument("--target_w", type=int, default=320)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--active_lines", type=int, default=8)
    p.add_argument("--pv_eps_probe", type=float, default=5e-2)
    p.add_argument("--sens_refresh_every", type=int, default=4)
    p.add_argument("--sens_blend", type=float, default=0.5)
    p.add_argument("--methods", nargs="+",
                   default=["pigdm_mc", "fakgd_mc", "pv", "pv_sens", "pv_sens_active"])
    args = p.parse_args()

    device = torch.device(args.device)
    target_shape = (args.target_h, args.target_w)

    print("Loading EDM model...")
    net = load_edm_model(args.checkpoint_dir, method="auto", device=args.device)
    denoiser = EDMDenoiser(net, device=args.device)

    sigma_schedule = edm_sigma_schedule(args.num_steps).to(device)
    print(f"T={args.num_steps}, σ_max={sigma_schedule[0]:.3f}, σ_min={sigma_schedule[-1]:.5f}")

    h5_files = sorted(Path(args.data_path).glob("*.h5"))[: args.num_volumes]
    print(f"Volumes: {[f.name for f in h5_files]}")

    rows = []
    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            n_sl = f["kspace"].shape[0]
        mid = n_sl // 2
        slice_indices = list(range(max(0, mid - args.num_slices // 2),
                                   min(n_sl, mid - args.num_slices // 2 + args.num_slices)))

        for sl in slice_indices:
            print(f"\n{'='*70}\n{h5_file.name}  slice {sl}\n{'='*70}")
            mc_k_clean, rss_gt = load_multicoil(str(h5_file), sl, target_shape)
            mc_k_clean = mc_k_clean.to(device); rss_gt = rss_gt.to(device)
            Nc, H, W = mc_k_clean.shape

            sigma_c = estimate_noise_per_coil(mc_k_clean).to(device) * args.noise_scale
            gen = torch.Generator(device="cpu").manual_seed(args.seed + sl)
            n_re = torch.randn(mc_k_clean.shape, generator=gen).to(device)
            n_im = torch.randn(mc_k_clean.shape, generator=gen).to(device)
            noise = sigma_c.sqrt().view(Nc, 1, 1) * (n_re + 1j * n_im) / np.sqrt(2)
            mc_k_noisy = mc_k_clean + noise.to(mc_k_clean.dtype)

            mask = create_mask(W, args.center_fraction, args.acceleration, seed=42).to(device)
            mask = mask.expand(H, -1)
            mc_y = mask * mc_k_noisy

            sens = estimate_sens_maps_lowres(mc_k_noisy,
                                              center_fraction=args.center_fraction).to(device)
            sense_oracle = sense_combine(ifft2c(mc_k_clean), sens)
            sense_op = MultiCoilSENSE(mask, sens, sigma_c).to(device)

            mean_noise = sigma_c.mean().item()
            sigma_i_sq_mc = torch.full((H, W), float(mean_noise),
                                        dtype=torch.float32, device=device)

            radius_grid = build_radius_grid(H, W).to(device)
            r_max = radius_grid.max().item()
            r_acs = args.center_fraction * r_max

            results = {}

            if "pigdm_mc" in args.methods:
                t0 = time.time()
                print("  (1) ΠGDM-MC...")
                r = run_pigdm_mc(y_mc=mc_y, sense_op=sense_op,
                                 sigma_schedule=sigma_schedule,
                                 denoiser_fn=denoiser, sigma_y=None, seed=args.seed)
                results["pigdm_mc"] = r["recon"]
                print(f"    PSNR vs RSS={_psnr(rss_gt, r['recon'].abs()):.3f}   "
                      f"vs SENSE={_psnr(sense_oracle.abs(), r['recon'].abs()):.3f}   "
                      f"({time.time()-t0:.1f}s)")

            if "fakgd_mc" in args.methods:
                t0 = time.time()
                print("  (2) FA-KGD-MC (FPDC, no PV)...")
                r = run_fakgd_mc(y_mc=mc_y, sense_op=sense_op,
                                 sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                                 sigma_i_sq_init=sigma_i_sq_mc, r_acs=r_acs, r_max=r_max,
                                 beta_fpdc=1.0, alpha_ema=0.95, gamma=0.0,
                                 m_step_mode="clamp", seed=args.seed)
                results["fakgd_mc"] = r["recon"]
                print(f"    PSNR vs RSS={_psnr(rss_gt, r['recon'].abs()):.3f}   "
                      f"vs SENSE={_psnr(sense_oracle.abs(), r['recon'].abs()):.3f}   "
                      f"({time.time()-t0:.1f}s)")

            if "pv" in args.methods:
                t0 = time.time()
                print("  (3) FA-KGD-MC + PV gate (A)...")
                r = run_fakgd_mc_pv(y_mc=mc_y, sense_op=sense_op,
                                    sigma_schedule=sigma_schedule,
                                    denoiser_fn=denoiser,
                                    sigma_i_sq_init=sigma_i_sq_mc,
                                    use_pv_gate=True,
                                    pv_eps_probe=args.pv_eps_probe,
                                    refine_sens=False, active_lines=0,
                                    seed=args.seed)
                results["pv"] = r["recon"]
                print(f"    PSNR vs RSS={_psnr(rss_gt, r['recon'].abs()):.3f}   "
                      f"vs SENSE={_psnr(sense_oracle.abs(), r['recon'].abs()):.3f}   "
                      f"({time.time()-t0:.1f}s)")

            if "pv_sens" in args.methods:
                t0 = time.time()
                print("  (4) FA-KGD-MC + PV + in-loop sens (A+B)...")
                r = run_fakgd_mc_pv(y_mc=mc_y, sense_op=sense_op,
                                    sigma_schedule=sigma_schedule,
                                    denoiser_fn=denoiser,
                                    sigma_i_sq_init=sigma_i_sq_mc,
                                    use_pv_gate=True,
                                    pv_eps_probe=args.pv_eps_probe,
                                    refine_sens=True,
                                    sens_refresh_every=args.sens_refresh_every,
                                    sens_blend=args.sens_blend,
                                    active_lines=0,
                                    seed=args.seed)
                results["pv_sens"] = r["recon"]
                print(f"    PSNR vs RSS={_psnr(rss_gt, r['recon'].abs()):.3f}   "
                      f"vs SENSE={_psnr(sense_oracle.abs(), r['recon'].abs()):.3f}   "
                      f"({time.time()-t0:.1f}s)")

            if "pv_sens_active" in args.methods:
                t0 = time.time()
                print(f"  (5) FA-KGD-MC + PV + sens + active ({args.active_lines} lines, A+B+D)...")
                r = run_fakgd_mc_pv(y_mc=mc_y, sense_op=sense_op,
                                    sigma_schedule=sigma_schedule,
                                    denoiser_fn=denoiser,
                                    sigma_i_sq_init=sigma_i_sq_mc,
                                    use_pv_gate=True,
                                    pv_eps_probe=args.pv_eps_probe,
                                    refine_sens=True,
                                    sens_refresh_every=args.sens_refresh_every,
                                    sens_blend=args.sens_blend,
                                    active_lines=args.active_lines,
                                    y_mc_full=mc_k_noisy,
                                    seed=args.seed)
                results["pv_sens_active"] = r["recon"]
                print(f"    PSNR vs RSS={_psnr(rss_gt, r['recon'].abs()):.3f}   "
                      f"vs SENSE={_psnr(sense_oracle.abs(), r['recon'].abs()):.3f}   "
                      f"({time.time()-t0:.1f}s)")

            row = dict(vol=h5_file.name, sl=sl)
            for name, recon in results.items():
                row[f"{name}_rss"] = _psnr(rss_gt, recon.abs())
                row[f"{name}_sense"] = _psnr(sense_oracle.abs(), recon.abs())
            rows.append(row)

    print(f"\n{'='*70}\nAGGREGATE  ({len(rows)} slices)\n{'='*70}")
    arr = lambda k: np.array([r.get(k, np.nan) for r in rows])
    methods_present = [m for m in args.methods if any(f"{m}_rss" in r for r in rows)]
    print(f"\n  {'method':>20s}  {'PSNR_RSS':>12s}  {'PSNR_SENSE':>12s}")
    for m in methods_present:
        v_rss = arr(f"{m}_rss"); v_sense = arr(f"{m}_sense")
        print(f"  {m:>20s}  {v_rss.mean():>8.3f}±{v_rss.std():.3f}  {v_sense.mean():>8.3f}±{v_sense.std():.3f}")
    if "pigdm_mc" in methods_present:
        base_rss = arr("pigdm_mc_rss"); base_sense = arr("pigdm_mc_sense")
        print(f"\n  ΔPSNR vs ΠGDM-MC   (RSS / SENSE)")
        for m in methods_present:
            if m == "pigdm_mc": continue
            d_rss = (arr(f"{m}_rss") - base_rss).mean()
            d_sense = (arr(f"{m}_sense") - base_sense).mean()
            print(f"    {m:>20s}  {d_rss:+8.3f} dB  /  {d_sense:+8.3f} dB")


if __name__ == "__main__":
    main()
