"""Reconstruct fastMRI slices using ΠGDM and FA-KGD+FPDC.

Works with either the oracle denoiser (for validation) or a real EDM model.

Usage:
    # Oracle mode (no GPU / checkpoint needed):
    python scripts/reconstruct.py --mode oracle --data_path data/singlecoil_test \
        --num_slices 5 --acceleration 4 --output_dir outputs/oracle_test

    # EDM mode:
    python scripts/reconstruct.py --mode edm --checkpoint_dir checkpoints/edm_R1 \
        --data_path data/singlecoil_test --num_slices 5 --acceleration 4 \
        --output_dir outputs/edm_test
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.mri_forward import fft2c, ifft2c, create_mask, build_radius_grid
from src.samplers.schedules import ddpm_sigma_schedule, edm_sigma_schedule
from src.samplers.pigdm import run_pigdm
from src.samplers.fakgd import run_fakgd
from src.models.edm_loader import OracleDenoiser, load_edm_model, EDMDenoiser


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Center crop the last two dimensions."""
    H, W = x.shape[-2], x.shape[-1]
    sh = (H - h) // 2
    sw = (W - w) // 2
    return x[..., sh : sh + h, sw : sw + w]


def load_slice(h5_path: str, slice_idx: int, target_shape: tuple | None = None) -> torch.Tensor:
    """Load a single slice from an HDF5 file, return complex image [H, W].

    Supports three data formats:
    1. Singlecoil: kspace [num_slices, H, W] → ifft2c
    2. Multicoil fully-sampled: kspace [num_slices, num_coils, H, W] → RSS
    3. Multicoil with reconstruction_rss: uses precomputed RSS ground truth
       (for val/test sets where k-space may be undersampled)

    If target_shape is provided, center-crops the image to that size.
    """
    with h5py.File(h5_path, "r") as f:
        if "reconstruction_rss" in f:
            # Val/test data: use precomputed RSS ground truth (real-valued)
            rss = f["reconstruction_rss"][slice_idx]  # (H, W) float
            img = torch.from_numpy(rss).to(torch.complex64)
        else:
            kspace = f["kspace"][slice_idx]  # (H, W) or (num_coils, H, W)
            kspace = torch.from_numpy(kspace)
            if kspace.dim() == 3:
                # Multicoil: RSS combination
                coil_imgs = ifft2c(kspace)
                rss = torch.sqrt((coil_imgs.abs() ** 2).sum(dim=0))
                img = rss.to(torch.complex64)
            else:
                img = ifft2c(kspace)

    if target_shape is not None:
        img = _center_crop(img, target_shape[0], target_shape[1])
    return img


def normalize_image(img: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Normalize complex image to [0, 1] magnitude range. Returns (img, scale)."""
    scale = img.abs().max().item()
    if scale < 1e-12:
        scale = 1.0
    return img / scale, scale


def make_freq_dependent_noise(H: int, W: int, sigma_base: float = 0.001, beta_noise: float = 5.0) -> torch.Tensor:
    """Create frequency-dependent noise std map (higher noise at high frequencies).

    σ²(r) = sigma_base * (1 + beta_noise * (r/r_max)²)
    This gives beta_noise+1 variation from center to edge.
    """
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    radius = torch.sqrt((gy - cy).float() ** 2 + (gx - cx).float() ** 2)
    r_norm = radius / radius.max()
    sigma_sq = sigma_base * (1 + beta_noise * r_norm ** 2)
    return sigma_sq


def add_measurement_noise(y_clean: torch.Tensor, sigma_sq_map: torch.Tensor) -> torch.Tensor:
    """Add frequency-dependent complex Gaussian noise to k-space."""
    sigma = sigma_sq_map.sqrt()
    noise = sigma * (torch.randn_like(y_clean.real) + 1j * torch.randn_like(y_clean.real)) / np.sqrt(2)
    return y_clean + noise


def run_reconstruction(args):
    """Main reconstruction loop."""
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect HDF5 files
    data_path = Path(args.data_path)
    h5_files = sorted(data_path.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_path}")

    print(f"Found {len(h5_files)} volumes in {data_path}")

    # Sigma schedule
    if args.schedule == "ddpm":
        sigma_schedule = ddpm_sigma_schedule(args.num_steps).to(device)
    else:
        sigma_schedule = edm_sigma_schedule(
            args.num_steps,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        ).to(device)

    print(f"Schedule: {args.schedule}, T={args.num_steps}, "
          f"σ_max={sigma_schedule[0]:.4f}, σ_min={sigma_schedule[-1]:.6f}")

    # --- Load model once (outside slice loop) ---
    edm_denoiser = None
    if args.mode == "edm":
        print("Loading EDM model...")
        net = load_edm_model(args.checkpoint_dir, method="auto", device=args.device)
        edm_denoiser = EDMDenoiser(net, device=args.device)
        print(f"  Model: {type(net).__name__}, {sum(p.numel() for p in net.parameters()):,} params")
        print(f"  Resolution: {net.img_resolution}, Channels: {net.img_channels}")

    # Results accumulator
    all_results = []
    slices_done = 0

    # Resolve M-step mode
    m_step_mode = args.m_step_mode
    if m_step_mode == "auto":
        m_step_mode = "full" if args.mode == "oracle" else "clamp"
    print(f"M-step mode: {m_step_mode}")

    for h5_file in h5_files:
        if slices_done >= args.num_slices:
            break

        with h5py.File(h5_file, "r") as f:
            key = "reconstruction_rss" if "reconstruction_rss" in f else "kspace"
            num_slices_in_vol = f[key].shape[0]

        # Pick middle slices (most informative anatomy)
        mid = num_slices_in_vol // 2
        slice_indices = range(max(0, mid - 2), min(num_slices_in_vol, mid + 3))

        for sl in slice_indices:
            if slices_done >= args.num_slices:
                break

            print(f"\n{'='*60}")
            print(f"Volume: {h5_file.name}, Slice: {sl}")
            print(f"{'='*60}")

            # Load and normalize (crop to model resolution if needed)
            target_shape = None
            if args.mode == "edm" and args.target_resolution:
                target_shape = tuple(args.target_resolution)
            x_gt = load_slice(str(h5_file), sl, target_shape=target_shape).to(device)
            x_gt, scale = normalize_image(x_gt)
            H, W = x_gt.shape

            # Create mask and measurements
            mask_1d = create_mask(
                W,
                center_fraction=args.center_fraction,
                acceleration=args.acceleration,
                seed=args.mask_seed,
            ).to(device)
            mask = mask_1d.expand(H, -1)

            y_full = fft2c(x_gt)

            # Add frequency-dependent measurement noise (matches NB04 setup)
            true_sigma_sq = make_freq_dependent_noise(
                H, W, sigma_base=args.sigma_base, beta_noise=args.beta_noise,
            ).to(device)
            y_noisy = add_measurement_noise(y_full, true_sigma_sq)
            y = mask * y_noisy

            # Noise initialization
            if m_step_mode == "full":
                # 2× overestimate tests EM convergence
                sigma_i_sq_init = 2.0 * true_sigma_sq
            else:
                # Fixed or clamped: use best estimate directly
                sigma_i_sq_init = true_sigma_sq.clone()

            # FPDC parameters
            radius_grid = build_radius_grid(H, W).to(device)
            r_max = radius_grid.max().item()
            r_acs = args.center_fraction * r_max

            # --- Set up denoiser ---
            if args.mode == "oracle":
                denoiser = OracleDenoiser(x_gt, eta=args.oracle_eta)
            elif args.mode == "edm":
                denoiser = edm_denoiser
            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            # Estimate sigma_y for ΠGDM (isotropic — mean of true noise variance)
            sigma_y_iso = np.sqrt(true_sigma_sq.mean().item())

            # --- Run ΠGDM ---
            print(f"\nRunning ΠGDM (T={args.num_steps}, σ_y={sigma_y_iso:.4e})...")
            t0 = time.time()
            pigdm_result = run_pigdm(
                y=y, mask=mask, sigma_schedule=sigma_schedule,
                denoiser_fn=denoiser, sigma_y=sigma_y_iso,
                x_gt=x_gt, seed=args.seed,
            )
            pigdm_time = time.time() - t0
            pigdm_psnr = pigdm_result["psnr_trajectory"][-1] if pigdm_result["psnr_trajectory"] else 0
            print(f"  ΠGDM: PSNR = {pigdm_psnr:.2f} dB  ({pigdm_time:.1f}s)")

            # --- Run FA-KGD + FPDC ---
            print(f"\nRunning FA-KGD+FPDC (β={args.beta_fpdc}, α={args.alpha_ema}, γ={args.gamma}, m_step={m_step_mode})...")
            t0 = time.time()
            fakgd_result = run_fakgd(
                y=y, mask=mask, sigma_schedule=sigma_schedule,
                denoiser_fn=denoiser, sigma_i_sq_init=sigma_i_sq_init,
                r_acs=r_acs, r_max=r_max, beta_fpdc=args.beta_fpdc,
                alpha_ema=args.alpha_ema, gamma=args.gamma,
                m_step_mode=m_step_mode,
                x_gt=x_gt, seed=args.seed, return_diagnostics=True,
            )
            fakgd_time = time.time() - t0
            fakgd_psnr = fakgd_result["psnr_trajectory"][-1] if fakgd_result["psnr_trajectory"] else 0
            print(f"  FA-KGD+FPDC: PSNR = {fakgd_psnr:.2f} dB  ({fakgd_time:.1f}s)")
            print(f"  Δ PSNR = {fakgd_psnr - pigdm_psnr:+.2f} dB")

            # Store results
            slice_result = {
                "volume": h5_file.name,
                "slice": sl,
                "pigdm_psnr": pigdm_psnr,
                "fakgd_psnr": fakgd_psnr,
                "delta_psnr": fakgd_psnr - pigdm_psnr,
                "pigdm_time": pigdm_time,
                "fakgd_time": fakgd_time,
            }
            all_results.append(slice_result)

            # Save reconstructions
            save_dir = os.path.join(args.output_dir, h5_file.stem)
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "pigdm_recon": pigdm_result["recon"].cpu(),
                "fakgd_recon": fakgd_result["recon"].cpu(),
                "pigdm_psnr_traj": pigdm_result["psnr_trajectory"],
                "fakgd_psnr_traj": fakgd_result["psnr_trajectory"],
                "x_gt": x_gt.cpu(),
                "mask": mask.cpu(),
                "scale": scale,
            }, os.path.join(save_dir, f"slice_{sl:03d}.pt"))

            slices_done += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({slices_done} slices, R={args.acceleration})")
    print(f"{'='*60}")
    pigdm_psnrs = [r["pigdm_psnr"] for r in all_results]
    fakgd_psnrs = [r["fakgd_psnr"] for r in all_results]
    deltas = [r["delta_psnr"] for r in all_results]
    print(f"  ΠGDM  mean PSNR: {np.mean(pigdm_psnrs):.2f} ± {np.std(pigdm_psnrs):.2f} dB")
    print(f"  FA-KGD mean PSNR: {np.mean(fakgd_psnrs):.2f} ± {np.std(fakgd_psnrs):.2f} dB")
    print(f"  Δ PSNR:           {np.mean(deltas):+.2f} ± {np.std(deltas):.2f} dB")

    # Save summary
    summary_path = os.path.join(args.output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": vars(args),
            "per_slice": all_results,
            "summary": {
                "pigdm_mean_psnr": float(np.mean(pigdm_psnrs)),
                "fakgd_mean_psnr": float(np.mean(fakgd_psnrs)),
                "delta_psnr_mean": float(np.mean(deltas)),
                "delta_psnr_std": float(np.std(deltas)),
                "num_slices": slices_done,
            },
        }, f, indent=2)
    print(f"\nResults saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="MRI reconstruction with ΠGDM and FA-KGD+FPDC")

    # Mode
    parser.add_argument("--mode", type=str, default="oracle", choices=["oracle", "edm"],
                        help="Denoiser mode: oracle (needs ground truth) or edm (real model)")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to HDF5 data directory")
    parser.add_argument("--num_slices", type=int, default=5, help="Number of slices to reconstruct")

    # MRI acquisition
    parser.add_argument("--acceleration", type=int, default=4, help="Acceleration factor")
    parser.add_argument("--center_fraction", type=float, default=0.08, help="ACS center fraction")
    parser.add_argument("--mask_seed", type=int, default=42, help="Mask random seed")

    # Diffusion schedule
    parser.add_argument("--schedule", type=str, default="ddpm", choices=["ddpm", "edm"],
                        help="Noise schedule type")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="Min sigma (EDM schedule)")
    parser.add_argument("--sigma_max", type=float, default=80.0, help="Max sigma (EDM schedule)")

    # FA-KGD parameters
    parser.add_argument("--beta_fpdc", type=float, default=1.0, help="FPDC schedule exponent")
    parser.add_argument("--alpha_ema", type=float, default=0.95, help="EMA smoothing for M-step")
    parser.add_argument("--gamma", type=float, default=0.0, help="Bias correction (0=none)")
    parser.add_argument("--m_step_mode", type=str, default="auto",
                        choices=["full", "clamp", "off", "auto"],
                        help="M-step mode: full (oracle), clamp (real model), off, auto (picks by mode)")

    # Measurement noise
    parser.add_argument("--sigma_base", type=float, default=0.001, help="Base noise variance")
    parser.add_argument("--beta_noise", type=float, default=5.0, help="Freq-dependent noise slope")

    # Oracle
    parser.add_argument("--oracle_eta", type=float, default=0.1, help="Oracle denoiser noise level")

    # EDM model
    parser.add_argument("--checkpoint_dir", type=str, default="", help="EDM checkpoint directory")
    parser.add_argument("--target_resolution", type=int, nargs=2, default=None,
                        help="Target image resolution [H W] for k-space cropping (e.g. 384 320)")

    # General
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--output_dir", type=str, default="outputs/recon", help="Output directory")

    args = parser.parse_args()
    run_reconstruction(args)


if __name__ == "__main__":
    main()
