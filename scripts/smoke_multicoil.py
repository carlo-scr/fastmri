"""Smoke test for multi-coil SENSE FA-KGD on fastMRI brain.

Compares four pipelines on the same slices and noise realisation:

    1. ΠGDM, single-coil RSS  (current baseline)
    2. FA-KGD, single-coil RSS
    3. ΠGDM, multi-coil SENSE
    4. FA-KGD, multi-coil SENSE  ← the new method

PSNR is computed against the fastMRI `reconstruction_rss` ground truth
(magnitude). The multi-coil reconstructions are taken as |x_recon|.

Defaults are small (T=20, 3 slices, 1 volume) so the script runs in a
few minutes on CPU. Pass --num_slices 5 / --num_steps 50 for a heavier
check.
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

from src.samplers.mri_forward import (
    fft2c, ifft2c, create_mask, build_radius_grid,
)
from src.samplers.schedules import edm_sigma_schedule
from src.samplers.pigdm import run_pigdm
from src.samplers.fakgd import run_fakgd
from src.samplers.acs import estimate_sigma_sq_pooled
from src.samplers.sense import (
    estimate_sens_maps_lowres, estimate_noise_per_coil, MultiCoilSENSE,
    sense_combine,
)
from src.samplers.multicoil import run_pigdm_mc, run_fakgd_mc
from src.models.edm_loader import load_edm_model, EDMDenoiser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    pad_h = max(0, h - H)
    pad_w = max(0, w - W)
    if pad_h or pad_w:
        ph0, pw0 = pad_h // 2, pad_w // 2
        ph1, pw1 = pad_h - ph0, pad_w - pw0
        x = torch.nn.functional.pad(x, (pw0, pw1, ph0, ph1))
        H, W = x.shape[-2], x.shape[-1]
    sh = (H - h) // 2
    sw = (W - w) // 2
    return x[..., sh : sh + h, sw : sw + w]


def _psnr(gt_mag: torch.Tensor, recon_mag: torch.Tensor) -> float:
    H = min(gt_mag.shape[-2], 320)
    W = min(gt_mag.shape[-1], 320)
    gt = _center_crop(gt_mag, H, W)
    rc = _center_crop(recon_mag, H, W)
    data_range = gt.max()
    mse = ((gt - rc) ** 2).mean()
    if mse < 1e-12:
        return 100.0
    return (10 * torch.log10(data_range**2 / mse)).item()


def load_multicoil(h5_path: str, slice_idx: int, target_shape):
    """Return (mc_kspace cropped & normalized (Nc,H,W) complex,
              rss_gt (H,W) real, scale factor used).

    Normalization: divide by max(RSS) so the RSS image lives in [0,1]
    magnitude — this matches the existing single-coil pipeline and the EDM
    training distribution.
    """
    with h5py.File(h5_path, "r") as f:
        ks = torch.from_numpy(f["kspace"][slice_idx])  # (Nc, H_full, W_full)
    coil_imgs_full = ifft2c(ks)
    coil_imgs = _center_crop(coil_imgs_full, target_shape[0], target_shape[1])
    rss_recompute = torch.sqrt((coil_imgs.abs() ** 2).sum(dim=0))
    scale = float(rss_recompute.max())
    if scale < 1e-30:
        scale = 1.0
    coil_imgs = coil_imgs / scale
    rss_recompute = rss_recompute / scale
    mc_k = fft2c(coil_imgs)
    return mc_k, rss_recompute, scale


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/brain_val")
    p.add_argument("--checkpoint_dir", default="checkpoints/edm/supervised_R=1")
    p.add_argument("--num_volumes", type=int, default=1)
    p.add_argument("--num_slices", type=int, default=3)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--acceleration", type=int, default=4)
    p.add_argument("--center_fraction", type=float, default=0.08)
    p.add_argument("--target_h", type=int, default=384)
    p.add_argument("--target_w", type=int, default=320)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_scale", type=float, default=1.0,
                   help="multiplier on the per-coil σ_c^2 before adding "
                        "synthetic noise (1.0 = realistic level estimated "
                        "from the corner patches).")
    p.add_argument("--alpha_radial", type=float, default=0.0,
                   help="Radial-quadratic boost for FA-KGD-MC σ_i² init: "
                        "σ_i²(r) = σ_white * (1 + alpha * (r/r_max)^2). "
                        "Captures SENSE coil-map model error which grows at "
                        "high frequencies. 0 = flat (matches ΠGDM-MC).")
    p.add_argument("--beta_fpdc", type=float, default=1.0,
                   help="FPDC schedule exponent.")
    p.add_argument("--skip_sc", action="store_true",
                   help="Skip the single-coil baselines (saves ~70s/slice).")
    p.add_argument("--m_step_mode", choices=["clamp", "full", "off"],
                   default="clamp")
    p.add_argument("--m_step_start_frac", type=float, default=0.0)
    p.add_argument("--alpha_ema", type=float, default=0.95)
    args = p.parse_args()

    device = torch.device(args.device)
    target_shape = (args.target_h, args.target_w)

    # Load EDM
    print("Loading EDM model...")
    net = load_edm_model(args.checkpoint_dir, method="auto", device=args.device)
    denoiser = EDMDenoiser(net, device=args.device)

    sigma_schedule = edm_sigma_schedule(args.num_steps).to(device)
    print(f"EDM sigma schedule: T={args.num_steps}, σ_max={sigma_schedule[0]:.3f}, σ_min={sigma_schedule[-1]:.5f}")

    h5_files = sorted(Path(args.data_path).glob("*.h5"))[: args.num_volumes]
    print(f"Volumes: {[f.name for f in h5_files]}")

    rows = []  # one row per slice with all four results

    for h5_file in h5_files:
        with h5py.File(h5_file, "r") as f:
            n_sl = f["kspace"].shape[0]
        mid = n_sl // 2
        slice_indices = range(max(0, mid - args.num_slices // 2),
                              min(n_sl, mid - args.num_slices // 2 + args.num_slices))

        for sl in slice_indices:
            print(f"\n{'='*70}\n{h5_file.name}  slice {sl}\n{'='*70}")
            mc_k_clean, rss_gt, scale = load_multicoil(str(h5_file), sl, target_shape)
            mc_k_clean = mc_k_clean.to(device)
            rss_gt = rss_gt.to(device)
            Nc, H, W = mc_k_clean.shape

            # --- Estimate per-coil noise σ_c^2 from the CLEAN k-space corners
            # Then add a controlled extra noise term so the experiment is
            # repeatable across hardware (real fastMRI noise is small).
            sigma_c_clean = estimate_noise_per_coil(mc_k_clean).to(device)
            sigma_c_sq = sigma_c_clean * args.noise_scale  # treated as the true noise

            # Add Gaussian noise per coil
            gen = torch.Generator(device="cpu").manual_seed(args.seed + sl)
            n_re = torch.randn(mc_k_clean.shape, generator=gen).to(device)
            n_im = torch.randn(mc_k_clean.shape, generator=gen).to(device)
            sigma_c_sqrt = sigma_c_sq.sqrt().view(Nc, 1, 1)
            noise = sigma_c_sqrt * (n_re + 1j * n_im) / np.sqrt(2)
            mc_k_noisy = mc_k_clean + noise.to(mc_k_clean.dtype)

            # --- Mask
            mask_1d = create_mask(W, args.center_fraction, args.acceleration,
                                  seed=42).to(device)
            mask = mask_1d.expand(H, -1)

            mc_y = mask * mc_k_noisy

            # --- Sensitivity maps from ACS (use full mc_k_noisy ACS region —
            # in practice the ACS is fully sampled so this is faithful).
            sens = estimate_sens_maps_lowres(
                mc_k_noisy, center_fraction=args.center_fraction,
            ).to(device)

            # SENSE-combined oracle (no undersampling, no DC) — the natural
            # self-consistent ground truth for any SENSE-based reconstruction.
            coil_imgs_clean = ifft2c(mc_k_clean)
            sense_oracle = sense_combine(coil_imgs_clean, sens)

            # --- Build SENSE operator
            sense_op = MultiCoilSENSE(mask, sens, sigma_c_sq).to(device)

            # --- Build single-coil (RSS) inputs (for the SC baselines).
            # We collapse each multi-coil image to RSS and treat it as if it
            # were the original measurement, applying the same mask. This
            # mirrors the historical pipeline.
            coil_imgs_noisy = ifft2c(mc_k_noisy)
            rss_noisy = torch.sqrt((coil_imgs_noisy.abs() ** 2).sum(dim=0)).to(torch.complex64)
            sc_y_noisy = fft2c(rss_noisy)
            sc_y = mask * sc_y_noisy

            # σ²_i for SC FA-KGD: pooled across coils within the slice
            # (estimate_sigma_sq_pooled wants y of shape (Nz,H,W)).
            sigma_i_sq_sc = estimate_sigma_sq_pooled(
                mc_k_noisy, center_fraction=args.center_fraction,
            ).to(device)
            # For multi-coil FA-KGD: a flat per-frequency map (real fastMRI
            # k-space noise is white to first order). The coil-dependent
            # heteroscedasticity is captured separately by sigma_c_sq via
            # sig_ci_sq = sigma_i_sq * sig_c_norm.  Using the mean of the
            # per-coil corner variances as the flat baseline.
            mean_noise = sigma_c_sq.mean().item()
            sigma_i_sq_mc = torch.full((H, W), float(mean_noise),
                                        dtype=torch.float32, device=device)
            if args.alpha_radial > 0:
                rg = build_radius_grid(H, W).to(device)
                rmax = rg.max().clamp(min=1.0)
                sigma_i_sq_mc = sigma_i_sq_mc * (
                    1.0 + args.alpha_radial * (rg / rmax) ** 2
                )

            # FPDC params
            radius_grid = build_radius_grid(H, W).to(device)
            r_max = radius_grid.max().item()
            r_acs = args.center_fraction * r_max

            # ΠGDM σ_y for single-coil baseline
            sigma_y_sc = float(sigma_i_sq_sc.mean().sqrt().item())

            # ============================================================
            # Run all four
            # ============================================================
            t0 = time.time()
            print("  (1) ΠGDM single-coil RSS...")
            if args.skip_sc:
                psnr_pigdm_sc = float("nan")
                print("    [skipped]")
            else:
                r_pigdm_sc = run_pigdm(
                    y=sc_y, mask=mask, sigma_schedule=sigma_schedule,
                    denoiser_fn=denoiser, sigma_y=sigma_y_sc,
                    x_gt=rss_noisy, seed=args.seed,
                )
                psnr_pigdm_sc = _psnr(rss_gt, r_pigdm_sc["recon"].abs())
                print(f"    PSNR = {psnr_pigdm_sc:.3f} dB  ({time.time()-t0:.1f}s)")

            t0 = time.time()
            print("  (2) FA-KGD single-coil RSS (m_step=clamp)...")
            if args.skip_sc:
                psnr_fakgd_sc = float("nan")
                print("    [skipped]")
            else:
                r_fakgd_sc = run_fakgd(
                    y=sc_y, mask=mask, sigma_schedule=sigma_schedule,
                    denoiser_fn=denoiser, sigma_i_sq_init=sigma_i_sq_sc,
                    r_acs=r_acs, r_max=r_max, beta_fpdc=1.0,
                    alpha_ema=0.95, gamma=0.0, m_step_mode="clamp",
                    x_gt=rss_noisy, seed=args.seed,
                )
                psnr_fakgd_sc = _psnr(rss_gt, r_fakgd_sc["recon"].abs())
                print(f"    PSNR = {psnr_fakgd_sc:.3f} dB  ({time.time()-t0:.1f}s)")

            t0 = time.time()
            print("  (3) ΠGDM multi-coil SENSE...")
            r_pigdm_mc = run_pigdm_mc(
                y_mc=mc_y, sense_op=sense_op, sigma_schedule=sigma_schedule,
                denoiser_fn=denoiser, sigma_y=None,
                x_gt=None, seed=args.seed,
            )
            psnr_pigdm_mc = _psnr(rss_gt, r_pigdm_mc["recon"].abs())
            print(f"    PSNR = {psnr_pigdm_mc:.3f} dB  ({time.time()-t0:.1f}s)")

            t0 = time.time()
            print("  (4) FA-KGD multi-coil SENSE (m_step=clamp)...")
            r_fakgd_mc = run_fakgd_mc(
                y_mc=mc_y, sense_op=sense_op, sigma_schedule=sigma_schedule,
                denoiser_fn=denoiser, sigma_i_sq_init=sigma_i_sq_mc,
                r_acs=r_acs, r_max=r_max, beta_fpdc=args.beta_fpdc,
                alpha_ema=args.alpha_ema, gamma=0.0,
                m_step_mode=args.m_step_mode,
                m_step_start_frac=args.m_step_start_frac,
                x_gt=None, seed=args.seed,
            )
            psnr_fakgd_mc = _psnr(rss_gt, r_fakgd_mc["recon"].abs())
            print(f"    PSNR = {psnr_fakgd_mc:.3f} dB  ({time.time()-t0:.1f}s)")

            # ---- Also evaluate vs SENSE-oracle (the self-consistent target)
            psnr_pigdm_mc_sense = _psnr(sense_oracle.abs(), r_pigdm_mc["recon"].abs())
            psnr_fakgd_mc_sense = _psnr(sense_oracle.abs(), r_fakgd_mc["recon"].abs())

            print("  --- summary ---")
            print(f"    ΠGDM  SC: {psnr_pigdm_sc:.3f}   FA-KGD SC: {psnr_fakgd_sc:.3f}   Δ_SC = {psnr_fakgd_sc - psnr_pigdm_sc:+.3f}")
            print(f"    ΠGDM  MC: {psnr_pigdm_mc:.3f}   FA-KGD MC: {psnr_fakgd_mc:.3f}   Δ_MC = {psnr_fakgd_mc - psnr_pigdm_mc:+.3f}")
            print(f"    MC vs SC for ΠGDM:   {psnr_pigdm_mc - psnr_pigdm_sc:+.3f} dB")
            print(f"    MC vs SC for FA-KGD: {psnr_fakgd_mc - psnr_fakgd_sc:+.3f} dB")
            print(f"    Total uplift FA-KGD MC vs ΠGDM SC: {psnr_fakgd_mc - psnr_pigdm_sc:+.3f} dB")
            print(f"    [vs SENSE-oracle] ΠGDM MC: {psnr_pigdm_mc_sense:.3f}   FA-KGD MC: {psnr_fakgd_mc_sense:.3f}   Δ = {psnr_fakgd_mc_sense - psnr_pigdm_mc_sense:+.3f}")

            rows.append(dict(
                vol=h5_file.name, sl=sl,
                pigdm_sc=psnr_pigdm_sc, fakgd_sc=psnr_fakgd_sc,
                pigdm_mc=psnr_pigdm_mc, fakgd_mc=psnr_fakgd_mc,
                pigdm_mc_sense=psnr_pigdm_mc_sense,
                fakgd_mc_sense=psnr_fakgd_mc_sense,
            ))

    # Aggregate
    print(f"\n{'='*70}\nAGGREGATE  ({len(rows)} slices)\n{'='*70}")
    arr = lambda k: np.array([r[k] for r in rows])
    for k in ("pigdm_sc", "fakgd_sc", "pigdm_mc", "fakgd_mc",
              "pigdm_mc_sense", "fakgd_mc_sense"):
        v = arr(k)
        print(f"  {k:>16s}: {v.mean():.3f} ± {v.std():.3f} dB")
    print(f"  Δ FA-KGD over ΠGDM (SC, vs RSS):     {(arr('fakgd_sc') - arr('pigdm_sc')).mean():+.3f} dB")
    print(f"  Δ FA-KGD over ΠGDM (MC, vs RSS):     {(arr('fakgd_mc') - arr('pigdm_mc')).mean():+.3f} dB")
    print(f"  Δ FA-KGD over ΠGDM (MC, vs SENSE):   {(arr('fakgd_mc_sense') - arr('pigdm_mc_sense')).mean():+.3f} dB")
    print(f"  Δ MC over SC (ΠGDM):      {(arr('pigdm_mc') - arr('pigdm_sc')).mean():+.3f} dB")
    print(f"  Δ MC over SC (FA-KGD):    {(arr('fakgd_mc') - arr('fakgd_sc')).mean():+.3f} dB")
    print(f"  Δ FA-KGD MC over ΠGDM SC: {(arr('fakgd_mc') - arr('pigdm_sc')).mean():+.3f} dB")


if __name__ == "__main__":
    main()
