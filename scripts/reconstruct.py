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
from skimage.metrics import structural_similarity as ssim_fn

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.mri_forward import fft2c, ifft2c, create_mask, build_radius_grid
from src.samplers.schedules import ddpm_sigma_schedule, edm_sigma_schedule
from src.samplers.pigdm import run_pigdm
from src.samplers.fakgd import run_fakgd
from src.samplers.dps import run_dps
from src.samplers.adps import run_adps
from src.samplers.acs import (
    estimate_sigma_sq_per_slice,
    estimate_sigma_sq_pooled,
    estimate_sigma_sq_multicoil_acs,
)
from src.models.edm_loader import OracleDenoiser, load_edm_model, EDMDenoiser


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Center crop the last two dimensions; zero-pad if source is smaller.

    Some fastMRI brain volumes (e.g. certain AXFLAIR files) have an
    `reconstruction_rss` / k-space matrix narrower than the requested
    target_resolution. Plain slicing then yields a tensor that's smaller
    than (h, w) and crashes the denoiser U-Net deep down. Pad symmetrically
    in any dimension that's too small, then crop.
    """
    H, W = x.shape[-2], x.shape[-1]
    pad_h = max(0, h - H)
    pad_w = max(0, w - W)
    if pad_h or pad_w:
        # F.pad order is (W_left, W_right, H_top, H_bottom)
        ph0, pw0 = pad_h // 2, pad_w // 2
        ph1, pw1 = pad_h - ph0, pad_w - pw0
        x = torch.nn.functional.pad(x, (pw0, pw1, ph0, ph1))
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


def load_slice_multicoil_kspace(
    h5_path: str, slice_idx: int, target_shape: tuple | None = None
) -> torch.Tensor | None:
    """Load raw multicoil k-space for one slice and crop to target_shape.

    Returns complex `(Nc, H, W)` k-space, matched in resolution to the RSS
    image returned by `load_slice` (i.e., IFFT → center-crop → FFT).

    Returns `None` if the file lacks a `kspace` dataset (e.g., RSS-only test
    data) or if it is not multicoil.
    """
    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            return None
        ks = f["kspace"][slice_idx]  # expected (Nc, H, W) for multicoil val
        ks = torch.from_numpy(ks)
    if ks.dim() != 3:
        return None
    coil_imgs = ifft2c(ks)
    if target_shape is not None:
        coil_imgs = _center_crop(coil_imgs, target_shape[0], target_shape[1])
    return fft2c(coil_imgs)


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


def make_white_noise(H: int, W: int, sigma_base: float = 0.001) -> torch.Tensor:
    """Flat σ² map (white k-space noise, matching real fastMRI thermal noise)."""
    return torch.full((H, W), float(sigma_base), dtype=torch.float32)


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

        # Slice selection: either middle-5 (default) or whole volume
        if args.whole_volume:
            slice_indices = range(num_slices_in_vol)
        else:
            mid = num_slices_in_vol // 2
            slice_indices = range(max(0, mid - 2), min(num_slices_in_vol, mid + 3))

        # --- Per-volume noise pre-pass (needed for pooled ACS estimator) ---
        # We add noise to every slice with a deterministic, per-volume seed so
        # that the pooled estimator and the per-slice loop see *the same*
        # measurement noise.
        target_shape = None
        if args.target_resolution:
            target_shape = tuple(args.target_resolution)
        gen = torch.Generator(device="cpu").manual_seed(args.noise_seed + hash(h5_file.name) % (2**31))

        # Probe one slice for shape & build the noise variance map
        probe = load_slice(str(h5_file), 0, target_shape=target_shape)
        Hp, Wp = probe.shape
        if args.noise_model == "white":
            true_sigma_sq_vol = make_white_noise(Hp, Wp, sigma_base=args.sigma_base)
        else:
            true_sigma_sq_vol = make_freq_dependent_noise(
                Hp, Wp, sigma_base=args.sigma_base, beta_noise=args.beta_noise,
            )
        # Pre-load every k-space slice we'll use, with reproducible noise
        y_noisy_per_slice = {}
        scale_per_slice = {}
        x_gt_per_slice = {}
        for sl in slice_indices:
            x_gt_sl = load_slice(str(h5_file), sl, target_shape=target_shape)
            x_gt_sl, scale_sl = normalize_image(x_gt_sl)
            y_clean = fft2c(x_gt_sl)
            sigma = true_sigma_sq_vol.sqrt()
            noise_re = torch.randn(y_clean.shape, generator=gen)
            noise_im = torch.randn(y_clean.shape, generator=gen)
            y_noisy = y_clean + sigma * (noise_re + 1j * noise_im) / np.sqrt(2)
            y_noisy_per_slice[sl] = y_noisy
            scale_per_slice[sl] = scale_sl
            x_gt_per_slice[sl] = x_gt_sl

        # Pooled estimate: stack all noisy slice k-spaces into (Nz_used, H, W)
        sigma_pooled_vol = None
        if args.noise_init == "pooled_acs":
            y_stack = torch.stack([y_noisy_per_slice[s] for s in slice_indices], dim=0)
            sigma_pooled_vol = estimate_sigma_sq_pooled(
                y_stack, center_fraction=args.center_fraction,
            ).to(device)
            print(
                f"[{h5_file.name}] pooled ACS over {len(slice_indices)} slices, "
                f"mean σ²={sigma_pooled_vol.mean().item():.3e}"
            )

        # Multicoil-ACS estimate: per-volume, pooled over the first K edge
        # slices (edge slices contain less anatomy → cleaner corner patches).
        # Per-slice estimates are averaged after the (robust) min-of-corners
        # aggregation inside the estimator.
        sigma_multicoil_vol = None
        if args.noise_init == "multicoil_acs":
            slice_list = list(slice_indices)
            K = max(1, min(args.acs_pool_slices, len(slice_list)))
            pool_slices = slice_list[:K]
            estimates = []
            for sli in pool_slices:
                y_mc_clean = load_slice_multicoil_kspace(
                    str(h5_file), sli, target_shape=target_shape,
                )
                if y_mc_clean is None:
                    raise RuntimeError(
                        f"{h5_file.name}: no multicoil k-space available for "
                        "--noise_init multicoil_acs"
                    )
                sigma_mc = true_sigma_sq_vol.sqrt()
                gen_mc = torch.Generator(device="cpu").manual_seed(
                    args.noise_seed + 7919 + sli + hash(h5_file.name) % (2**31)
                )
                n_re = torch.randn(y_mc_clean.shape, generator=gen_mc)
                n_im = torch.randn(y_mc_clean.shape, generator=gen_mc)
                y_mc_noisy = y_mc_clean + sigma_mc * (n_re + 1j * n_im) / np.sqrt(2)
                est_map = estimate_sigma_sq_multicoil_acs(
                    y_mc_noisy, method="image_bg",
                    bg_patch_frac=args.bg_patch_frac,
                    center_fraction=args.center_fraction,
                    aggregator=args.acs_aggregator,
                )
                estimates.append(est_map)
            sigma_multicoil_vol = torch.stack(estimates, dim=0).mean(dim=0).to(device)
            true_mean = float(true_sigma_sq_vol.mean())
            est_mean = float(sigma_multicoil_vol.mean())
            print(
                f"[{h5_file.name}] multicoil-ACS σ²={est_mean:.3e} "
                f"(true={true_mean:.3e}, ratio={est_mean / max(true_mean, 1e-30):.3f}, "
                f"pooled K={K} slices, agg={args.acs_aggregator})"
            )

        for sl in slice_indices:
            if slices_done >= args.num_slices:
                break

            print(f"\n{'='*60}")
            print(f"Volume: {h5_file.name}, Slice: {sl}")
            print(f"{'='*60}")

            # Pull from the per-volume pre-pass
            x_gt = x_gt_per_slice[sl].to(device)
            scale = scale_per_slice[sl]
            y_noisy = y_noisy_per_slice[sl].to(device)
            H, W = x_gt.shape
            true_sigma_sq = true_sigma_sq_vol.to(device)

            # Mask
            mask_1d = create_mask(
                W,
                center_fraction=args.center_fraction,
                acceleration=args.acceleration,
                seed=args.mask_seed,
            ).to(device)
            mask = mask_1d.expand(H, -1)
            y = mask * y_noisy

            # σ²_i initialisation strategy
            if args.noise_init == "oracle":
                # True noise variance (cheating upper bound)
                if m_step_mode == "full":
                    sigma_i_sq_init = 2.0 * true_sigma_sq  # tests EM convergence
                else:
                    sigma_i_sq_init = true_sigma_sq.clone()
            elif args.noise_init == "per_slice_acs":
                sigma_i_sq_init = estimate_sigma_sq_per_slice(
                    y_noisy, center_fraction=args.center_fraction,
                ).to(device)
            elif args.noise_init == "pooled_acs":
                sigma_i_sq_init = sigma_pooled_vol  # one estimate per volume
            elif args.noise_init == "multicoil_acs":
                sigma_i_sq_init = sigma_multicoil_vol  # one estimate per volume
            else:
                raise ValueError(f"Unknown noise_init: {args.noise_init}")

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

            # --- Run DPS ---
            dps_result = None
            dps_time = 0.0
            if "dps" in args.methods:
                print(f"\nRunning DPS (T={args.num_steps}, ζ={args.dps_step_size})...")
                t0 = time.time()
                dps_result = run_dps(
                    y=y, mask=mask, sigma_schedule=sigma_schedule,
                    denoiser_fn=denoiser, step_size=args.dps_step_size,
                    x_gt=x_gt, seed=args.seed,
                )
                dps_time = time.time() - t0
                dps_psnr = dps_result["psnr_trajectory"][-1] if dps_result["psnr_trajectory"] else 0
                print(f"  DPS: PSNR = {dps_psnr:.2f} dB  ({dps_time:.1f}s)")
            else:
                dps_psnr = float("nan")

            # --- Run ADPS ---
            adps_result = None
            adps_time = 0.0
            if "adps" in args.methods:
                print(f"\nRunning ADPS (T={args.num_steps}, l_ss={args.adps_step_size}, S_churn={args.adps_s_churn})...")
                t0 = time.time()
                adps_result = run_adps(
                    y=y, mask=mask, sigma_schedule=sigma_schedule,
                    denoiser_fn=denoiser, step_size=args.adps_step_size,
                    s_churn=args.adps_s_churn,
                    x_gt=x_gt, seed=args.seed,
                )
                adps_time = time.time() - t0
                adps_psnr = adps_result["psnr_trajectory"][-1] if adps_result["psnr_trajectory"] else 0
                print(f"  ADPS: PSNR = {adps_psnr:.2f} dB  ({adps_time:.1f}s)")
            else:
                adps_psnr = float("nan")

            # --- Run ΠGDM ---
            pigdm_result = None
            pigdm_time = 0.0
            if "pigdm" in args.methods:
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
            else:
                pigdm_psnr = float("nan")

            # --- Run FA-KGD + FPDC ---
            fakgd_result = None
            fakgd_time = 0.0
            if "fakgd" in args.methods:
                print(f"\nRunning FA-KGD+FPDC (β={args.beta_fpdc}, α={args.alpha_ema}, γ={args.gamma}, m_step={m_step_mode}, start_frac={args.m_step_start_frac})...")
                t0 = time.time()
                fakgd_result = run_fakgd(
                    y=y, mask=mask, sigma_schedule=sigma_schedule,
                    denoiser_fn=denoiser, sigma_i_sq_init=sigma_i_sq_init,
                    r_acs=r_acs, r_max=r_max, beta_fpdc=args.beta_fpdc,
                    alpha_ema=args.alpha_ema, gamma=args.gamma,
                    m_step_mode=m_step_mode,
                    m_step_start_frac=args.m_step_start_frac,
                    x_gt=x_gt, seed=args.seed, return_diagnostics=True,
                )
                fakgd_time = time.time() - t0
                fakgd_psnr = fakgd_result["psnr_trajectory"][-1] if fakgd_result["psnr_trajectory"] else 0
                print(f"  FA-KGD+FPDC: PSNR = {fakgd_psnr:.2f} dB  ({fakgd_time:.1f}s)")
                if not np.isnan(pigdm_psnr):
                    print(f"  Δ PSNR = {fakgd_psnr - pigdm_psnr:+.2f} dB")
            else:
                fakgd_psnr = float("nan")

            # Compute SSIM (only for methods that ran)
            gt_mag = x_gt.abs().cpu().numpy()
            data_range = gt_mag.max() - gt_mag.min()
            def _ssim_or_nan(res):
                if res is None:
                    return float("nan")
                m = res["recon"].abs().cpu().numpy()
                return float(ssim_fn(gt_mag, m, data_range=data_range))
            dps_ssim = _ssim_or_nan(dps_result)
            adps_ssim = _ssim_or_nan(adps_result)
            pigdm_ssim = _ssim_or_nan(pigdm_result)
            fakgd_ssim = _ssim_or_nan(fakgd_result)
            for name, val in (("DPS", dps_ssim), ("ADPS", adps_ssim),
                              ("ΠGDM", pigdm_ssim), ("FA-KGD", fakgd_ssim)):
                if not np.isnan(val):
                    print(f"  {name:>6s} SSIM = {val:.4f}")

            # Store results
            slice_result = {
                "volume": h5_file.name,
                "slice": sl,
                "dps_psnr": dps_psnr,
                "adps_psnr": adps_psnr,
                "pigdm_psnr": pigdm_psnr,
                "fakgd_psnr": fakgd_psnr,
                "delta_psnr": fakgd_psnr - pigdm_psnr,
                "dps_ssim": float(dps_ssim),
                "adps_ssim": float(adps_ssim),
                "pigdm_ssim": float(pigdm_ssim),
                "fakgd_ssim": float(fakgd_ssim),
                "delta_ssim": float(fakgd_ssim - pigdm_ssim),
                "dps_time": dps_time,
                "adps_time": adps_time,
                "pigdm_time": pigdm_time,
                "fakgd_time": fakgd_time,
            }
            all_results.append(slice_result)

            # Save reconstructions (only for methods that ran)
            save_dir = os.path.join(args.output_dir, h5_file.stem)
            os.makedirs(save_dir, exist_ok=True)
            payload: dict = {
                "x_gt": x_gt.cpu(),
                "mask": mask.cpu(),
                "scale": scale,
            }
            for name, res in (("dps", dps_result), ("adps", adps_result),
                              ("pigdm", pigdm_result), ("fakgd", fakgd_result)):
                if res is not None:
                    payload[f"{name}_recon"] = res["recon"].cpu()
                    payload[f"{name}_psnr_traj"] = res["psnr_trajectory"]
            torch.save(payload, os.path.join(save_dir, f"slice_{sl:03d}.pt"))

            slices_done += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({slices_done} slices, R={args.acceleration})")
    print(f"{'='*60}")
    dps_psnrs = [r["dps_psnr"] for r in all_results]
    adps_psnrs = [r["adps_psnr"] for r in all_results]
    pigdm_psnrs = [r["pigdm_psnr"] for r in all_results]
    fakgd_psnrs = [r["fakgd_psnr"] for r in all_results]
    deltas = [r["delta_psnr"] for r in all_results]
    dps_ssims = [r["dps_ssim"] for r in all_results]
    adps_ssims = [r["adps_ssim"] for r in all_results]
    pigdm_ssims = [r["pigdm_ssim"] for r in all_results]
    fakgd_ssims = [r["fakgd_ssim"] for r in all_results]
    delta_ssims = [r["delta_ssim"] for r in all_results]
    print(f"  DPS   mean PSNR: {np.mean(dps_psnrs):.2f} ± {np.std(dps_psnrs):.2f} dB")
    print(f"  ADPS  mean PSNR: {np.mean(adps_psnrs):.2f} ± {np.std(adps_psnrs):.2f} dB")
    print(f"  ΠGDM  mean PSNR: {np.mean(pigdm_psnrs):.2f} ± {np.std(pigdm_psnrs):.2f} dB")
    print(f"  FA-KGD mean PSNR: {np.mean(fakgd_psnrs):.2f} ± {np.std(fakgd_psnrs):.2f} dB")
    print(f"  Δ PSNR (FA-KGD vs ΠGDM): {np.mean(deltas):+.2f} ± {np.std(deltas):.2f} dB")
    print(f"  DPS   mean SSIM: {np.mean(dps_ssims):.4f} ± {np.std(dps_ssims):.4f}")
    print(f"  ADPS  mean SSIM: {np.mean(adps_ssims):.4f} ± {np.std(adps_ssims):.4f}")
    print(f"  ΠGDM  mean SSIM: {np.mean(pigdm_ssims):.4f} ± {np.std(pigdm_ssims):.4f}")
    print(f"  FA-KGD mean SSIM: {np.mean(fakgd_ssims):.4f} ± {np.std(fakgd_ssims):.4f}")
    print(f"  Δ SSIM:           {np.mean(delta_ssims):+.4f} ± {np.std(delta_ssims):.4f}")

    # --- Per-volume aggregation (mean over slices, then mean over volumes) ---
    from collections import defaultdict
    per_vol = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        for k in ("dps_psnr", "adps_psnr", "pigdm_psnr", "fakgd_psnr",
                  "dps_ssim", "adps_ssim", "pigdm_ssim", "fakgd_ssim"):
            per_vol[r["volume"]][k].append(r[k])
    vol_means = {
        k: [float(np.mean(per_vol[v][k])) for v in per_vol]
        for k in ("dps_psnr", "adps_psnr", "pigdm_psnr", "fakgd_psnr",
                  "dps_ssim", "adps_ssim", "pigdm_ssim", "fakgd_ssim")
    }
    print(f"\nPer-volume aggregation ({len(per_vol)} volumes):")
    for k in ("dps_psnr", "adps_psnr", "pigdm_psnr", "fakgd_psnr"):
        vals = vol_means[k]
        print(f"  {k:>12s}: {np.mean(vals):.2f} ± {np.std(vals):.2f} dB")
    for k in ("dps_ssim", "adps_ssim", "pigdm_ssim", "fakgd_ssim"):
        vals = vol_means[k]
        print(f"  {k:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump({
            "config": vars(args),
            "per_slice": all_results,
            "summary": {
                "dps_mean_psnr": float(np.mean(dps_psnrs)),
                "adps_mean_psnr": float(np.mean(adps_psnrs)),
                "pigdm_mean_psnr": float(np.mean(pigdm_psnrs)),
                "fakgd_mean_psnr": float(np.mean(fakgd_psnrs)),
                "delta_psnr_mean": float(np.mean(deltas)),
                "delta_psnr_std": float(np.std(deltas)),
                "dps_mean_ssim": float(np.mean(dps_ssims)),
                "adps_mean_ssim": float(np.mean(adps_ssims)),
                "pigdm_mean_ssim": float(np.mean(pigdm_ssims)),
                "fakgd_mean_ssim": float(np.mean(fakgd_ssims)),
                "delta_ssim_mean": float(np.mean(delta_ssims)),
                "delta_ssim_std": float(np.std(delta_ssims)),
                "num_slices": slices_done,
            },
            "per_volume": {
                v: {k: float(np.mean(per_vol[v][k])) for k in vol_means}
                for v in per_vol
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
    parser.add_argument("--m_step_start_frac", type=float, default=1.0,
                        help="Only run M-step once σ_t < frac*σ_max. "
                             "Default 1.0 = run from step 0 (original behaviour). "
                             "Use e.g. 0.5 with --m_step_mode full + flat init "
                             "(multicoil_acs) to recover frequency structure "
                             "without denoiser-error contamination at high σ_t.")

    # Measurement noise
    parser.add_argument("--sigma_base", type=float, default=0.001, help="Base noise variance")
    parser.add_argument("--beta_noise", type=float, default=5.0, help="Freq-dependent noise slope")
    parser.add_argument("--noise_seed", type=int, default=12345,
                        help="Per-volume noise seed (mixed with file name)")
    parser.add_argument("--noise_init", type=str, default="oracle",
                        choices=["oracle", "per_slice_acs", "pooled_acs", "multicoil_acs"],
                        help="How to initialise σ²_i for FA-KGD: oracle (true), "
                             "per_slice_acs (one slice's ACS), pooled_acs (volume),"
                             " or multicoil_acs (within-slice multicoil background patch).")
    parser.add_argument("--noise_model", type=str, default="freq_dep",
                        choices=["freq_dep", "white"],
                        help="Synthetic measurement-noise model: freq_dep (legacy, "
                             "frequency-dependent) or white (flat, matches real fastMRI "
                             "thermal noise; required for multicoil_acs estimator).")
    parser.add_argument("--bg_patch_frac", type=float, default=0.08,
                        help="Background-patch fraction for multicoil_acs estimator.")
    parser.add_argument("--acs_pool_slices", type=int, default=3,
                        help="Number of (edge) slices to pool for multicoil_acs "
                             "estimator (default 3 → averages 3 per-slice estimates).")
    parser.add_argument("--acs_aggregator", type=str, default="min",
                        choices=["min", "mean", "median"],
                        help="Per-slice corner aggregator for multicoil_acs "
                             "(min is robust to signal leakage into one corner).")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["dps", "adps", "pigdm", "fakgd"],
                        choices=["dps", "adps", "pigdm", "fakgd"],
                        help="Which methods to run (default: all four).")
    parser.add_argument("--whole_volume", action="store_true",
                        help="If set, process every slice of every visited volume "
                             "(needed for clean per-volume metrics).")

    # DPS
    parser.add_argument("--dps_step_size", type=float, default=10.0, help="DPS gradient step size (zeta)")

    # ADPS
    parser.add_argument("--adps_step_size", type=float, default=10.0, help="ADPS likelihood step size (l_ss)")
    parser.add_argument("--adps_s_churn", type=float, default=0.0, help="ADPS stochastic churn (S_churn)")

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
