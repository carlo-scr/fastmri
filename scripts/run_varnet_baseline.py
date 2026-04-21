"""Evaluate pretrained E2E VarNet on the same brain slices used for diffusion experiments.

Brain data is multicoil → VarNet applies directly.
Knee data is singlecoil → VarNet not applicable (requires multicoil sensitivity estimation).

Usage:
    python scripts/run_varnet_baseline.py
"""

import sys
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

from fastmri.models import VarNet
from fastmri.data.transforms import center_crop


def create_equispaced_mask(num_cols, center_fraction, acceleration, seed=42):
    """Create equispaced mask matching the mask used in diffusion experiments."""
    rng = np.random.RandomState(seed)
    num_low = int(round(num_cols * center_fraction))
    # Equispaced lines
    mask = np.zeros(num_cols, dtype=bool)
    # Offset by random amount for equispaced lines
    offset = rng.randint(0, acceleration)
    mask[offset::acceleration] = True
    # Always include center
    center_start = (num_cols - num_low) // 2
    mask[center_start:center_start + num_low] = True
    return mask


def run_varnet_on_brain(model, vol_path, slices, acceleration, center_fraction=0.08, seed=42):
    """Run VarNet on brain multicoil data."""
    results = []

    for sl_idx in slices:
        with h5py.File(vol_path, 'r') as f:
            kspace_np = f['kspace'][sl_idx]      # (num_coils, H, W) complex64
            rss_gt = f['reconstruction_rss'][sl_idx]  # (H, W) float32

        num_coils, H, W = kspace_np.shape

        # Create mask
        mask_1d = create_equispaced_mask(W, center_fraction, acceleration, seed=seed)
        num_low = int(round(W * center_fraction))

        # Convert to torch
        kspace = torch.from_numpy(kspace_np)
        kspace_real = torch.stack([kspace.real, kspace.imag], dim=-1).unsqueeze(0).float()

        # Apply mask
        mask_torch = torch.from_numpy(mask_1d).bool().reshape(1, 1, 1, W, 1)
        masked_kspace = kspace_real * mask_torch.float()

        # Run VarNet
        t0 = time.time()
        with torch.no_grad():
            output = model(masked_kspace, mask_torch, num_low_frequencies=num_low)
        elapsed = time.time() - t0

        # Center crop to match ground truth
        crop_size = (rss_gt.shape[0], rss_gt.shape[1])
        output_cropped = center_crop(output, crop_size)[0].numpy()

        # Compute metrics
        gt = rss_gt
        recon = output_cropped
        data_range = gt.max() - gt.min()
        mse = np.mean((gt - recon) ** 2)
        psnr = 10 * np.log10(data_range ** 2 / mse)
        ssim_val = float(ssim_fn(gt, recon, data_range=data_range))

        print(f'  Brain sl{sl_idx}: VarNet PSNR={psnr:.2f} SSIM={ssim_val:.4f} ({elapsed:.1f}s)')
        results.append({
            'slice': sl_idx,
            'psnr': float(psnr),
            'ssim': ssim_val,
            'time': elapsed,
        })

    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    print(f'  MEAN: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}')
    return results


def main():
    # Load pretrained brain VarNet
    print("Loading pretrained E2E VarNet (brain)...")
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    sd = torch.load(
        'checkpoints/varnet/brain_leaderboard_state_dict.pt',
        map_location='cpu',
        weights_only=False,
    )
    model.load_state_dict(sd)
    model.eval()
    print(f"  VarNet loaded: {sum(p.numel() for p in model.parameters()):,} params")

    vol_path = 'data/brain_val/file_brain_AXT2_200_2000093.h5'
    slices = list(range(6, 11))  # slices 6-10

    all_results = {}

    # Brain R=4
    print("\n=== Brain R=4 ===")
    all_results['brain_R4'] = run_varnet_on_brain(model, vol_path, slices, acceleration=4)

    # Brain R=8
    print("\n=== Brain R=8 ===")
    all_results['brain_R8'] = run_varnet_on_brain(model, vol_path, slices, acceleration=8)

    # Save results
    out_path = 'outputs/varnet_baseline_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n=== SUMMARY ===")
    for key, results in all_results.items():
        psnrs = [r['psnr'] for r in results]
        ssims = [r['ssim'] for r in results]
        print(f"{key}: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}")


if __name__ == '__main__':
    main()
