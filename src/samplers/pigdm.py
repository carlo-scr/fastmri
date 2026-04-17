"""ΠGDM (Pseudoinverse-Guided Diffusion Model) sampler for MRI reconstruction.

Implements the baseline method from Song et al. (2023).
Accepts any denoiser function: oracle or real score network.
"""

import torch
import numpy as np
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c


def run_pigdm(
    y: torch.Tensor,
    mask: torch.Tensor,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_y: float,
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> dict:
    """Run ΠGDM reconstruction.

    Args:
        y: Measured k-space (complex, [H, W]).
        mask: Undersampling mask (real, broadcastable to [H, W]).
        sigma_schedule: Noise levels descending from σ_max to σ_min, shape (T,).
        denoiser_fn: Callable(x_t, sigma_t) -> mu_theta. Takes complex image
            and noise level, returns denoised complex image.
        sigma_y: Isotropic measurement noise std (scalar).
        x_gt: Ground truth for PSNR tracking (optional).
        seed: Random seed.

    Returns:
        dict with keys: 'recon' (complex [H, W]), 'psnr_trajectory' (list).
    """
    torch.manual_seed(seed)
    T = len(sigma_schedule)
    H, W = y.shape[-2], y.shape[-1]
    device = y.device

    # Initialize from noise at sigma_max
    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)

    psnr_trajectory = []

    for step in range(T):
        sigma_t = sigma_schedule[step].item()

        # Denoiser prediction
        mu_theta = denoiser_fn(x_t, sigma_t)

        # K-space residual with isotropic Kalman gain
        mu_k = fft2c(mu_theta)
        residual = mask * (y - mu_k)
        K = sigma_t**2 / (sigma_t**2 + sigma_y**2)
        x_corrected = mu_theta + ifft2c(K * residual)

        # Reverse step
        if step < T - 1:
            sigma_next = sigma_schedule[step + 1].item()
            x_t = x_corrected + sigma_next * _complex_randn(H, W, device=device)
        else:
            x_t = x_corrected

        # Track PSNR
        if x_gt is not None:
            p = _psnr(x_gt, x_t)
            psnr_trajectory.append(p)

    return {"recon": x_t, "psnr_trajectory": psnr_trajectory}


class PIGDMSampler:
    """Object-oriented wrapper around run_pigdm for convenience."""

    def __init__(
        self,
        denoiser_fn: Callable,
        sigma_y: float = 1e-3,
    ):
        self.denoiser_fn = denoiser_fn
        self.sigma_y = sigma_y

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        sigma_schedule: torch.Tensor,
        x_gt: Optional[torch.Tensor] = None,
        seed: int = 0,
    ) -> dict:
        return run_pigdm(
            y=y,
            mask=mask,
            sigma_schedule=sigma_schedule,
            denoiser_fn=self.denoiser_fn,
            sigma_y=self.sigma_y,
            x_gt=x_gt,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _complex_randn(H: int, W: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return (torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)) / np.sqrt(2)


def _psnr(x_gt: torch.Tensor, x_recon: torch.Tensor) -> float:
    gt_mag = x_gt.abs()
    recon_mag = x_recon.abs()
    # Center crop to min size
    H, W = min(gt_mag.shape[-2], 320), min(gt_mag.shape[-1], 320)
    gt_mag = _center_crop(gt_mag, H, W)
    recon_mag = _center_crop(recon_mag, H, W)
    data_range = gt_mag.max()
    mse = ((gt_mag - recon_mag) ** 2).mean()
    if mse < 1e-12:
        return 100.0
    return (10 * torch.log10(data_range**2 / mse)).item()


def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    sh = (H - h) // 2
    sw = (W - w) // 2
    return x[..., sh : sh + h, sw : sw + w]
