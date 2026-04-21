"""DPS (Diffusion Posterior Sampling) sampler for MRI reconstruction.

Implements Chung et al. (2023) — "Diffusion Posterior Sampling for General
Noisy Inverse Problems".  The key difference from ΠGDM is that the data
consistency step uses a *gradient* of the likelihood through the denoised
estimate, rather than a closed-form pseudoinverse.

Reference:
    @inproceedings{chung2023dps,
      title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
      author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael T and
              Klasky, Marc L and Ye, Jong Chul},
      booktitle={ICLR},
      year={2023}
    }
"""

import torch
import numpy as np
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c


def run_dps(
    y: torch.Tensor,
    mask: torch.Tensor,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    step_size: float = 10.0,
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> dict:
    """Run DPS reconstruction.

    At each step t:
      1. Compute mu_theta = D(x_t, sigma_t)   (Tweedie denoised estimate)
      2. Compute likelihood loss  L = ||y - M * F(mu_theta)||^2
      3. Gradient step:  x_corrected = mu_theta - (zeta / ||residual||) * grad_x L
      4. Reverse diffusion step with noise injection

    DPS requires gradients through the denoiser. If the denoiser has a
    ``@torch.no_grad`` wrapper (e.g. EDMDenoiser), DPS bypasses it by
    calling the underlying network directly.  For non-differentiable
    denoisers (e.g. OracleDenoiser), falls back to an analytic gradient
    of the likelihood w.r.t. the denoised estimate directly.

    Step-size convention follows ADPS (Daras et al., 2024):
        correction = (zeta / ||y - A*mu||) * grad_{x_t} ||y - A*mu||^2

    Args:
        y: Measured k-space (complex, [H, W]).
        mask: Undersampling mask (real, broadcastable to [H, W]).
        sigma_schedule: Noise levels descending from sigma_max to sigma_min.
        denoiser_fn: Callable(x_t, sigma_t) -> mu_theta.
        step_size: Gradient step size (zeta). Scaled by 1/||residual||
            internally following the ADPS convention.
        x_gt: Ground truth for PSNR tracking (optional).
        seed: Random seed.

    Returns:
        dict with 'recon' and 'psnr_trajectory'.
    """
    torch.manual_seed(seed)
    T = len(sigma_schedule)
    H, W = y.shape[-2], y.shape[-1]
    device = y.device

    # Build a gradient-compatible denoiser wrapper.
    # EDMDenoiser uses @torch.no_grad, so we access its .net directly.
    _has_net = hasattr(denoiser_fn, "net")
    if _has_net:
        _net = denoiser_fn.net
        _dev = denoiser_fn.device if hasattr(denoiser_fn, "device") else device

    def _denoise_with_grad(x_complex: torch.Tensor, sigma_t: float) -> torch.Tensor:
        """Call the denoiser *with* gradient tracking."""
        if _has_net:
            # Replicate EDMDenoiser logic, but without @torch.no_grad
            x_real = torch.stack([x_complex.real, x_complex.imag], dim=0)
            x_real = x_real.unsqueeze(0).float().to(_dev)
            sigma = torch.tensor([sigma_t], dtype=torch.float32, device=_dev)
            denoised = _net(x_real, sigma)
            out = denoised[0, 0] + 1j * denoised[0, 1]
            return out.to(x_complex.device)
        else:
            # Oracle or other denoiser — just call it
            return denoiser_fn(x_complex, sigma_t)

    # Initialize from noise
    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)

    psnr_trajectory = []

    for step in range(T):
        sigma_t = sigma_schedule[step].item()

        # Enable gradient tracking for the likelihood step
        x_t_grad = x_t.clone().detach().requires_grad_(True)

        # Denoiser prediction (through the graph for gradient)
        mu_theta = _denoise_with_grad(x_t_grad, sigma_t)

        # Likelihood: ||y - mask * F(mu_theta)||^2
        residual_k = mask * (y - fft2c(mu_theta))
        loss = residual_k.abs().pow(2).sum()

        # Gradient w.r.t. x_t (through denoiser via autograd)
        try:
            grad = torch.autograd.grad(loss, x_t_grad)[0]
        except RuntimeError:
            # Fallback for non-differentiable denoisers (e.g. oracle):
            # use analytic gradient of ||y - MFμ||² w.r.t. μ directly
            grad = -2.0 * ifft2c(residual_k).detach()

        # ADPS-style step-size: zeta / ||residual||  (Daras et al., 2024)
        # This normalizes out the residual magnitude from the gradient,
        # making the effective step depend only on the Jacobian direction.
        residual_norm = loss.sqrt().item()
        scaled_step = step_size / max(residual_norm, 1e-8)

        # Move in direction that decreases ||y - Ax||^2
        mu_det = mu_theta.detach()
        x_corrected = mu_det - scaled_step * grad.detach()

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


class DPSSampler:
    """Object-oriented wrapper around run_dps."""

    def __init__(self, denoiser_fn: Callable, step_size: float = 1.0):
        self.denoiser_fn = denoiser_fn
        self.step_size = step_size

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        sigma_schedule: torch.Tensor,
        x_gt: Optional[torch.Tensor] = None,
        seed: int = 0,
    ) -> dict:
        return run_dps(
            y=y, mask=mask, sigma_schedule=sigma_schedule,
            denoiser_fn=self.denoiser_fn, step_size=self.step_size,
            x_gt=x_gt, seed=seed,
        )


# ---------------------------------------------------------------------------
# Helpers (shared with pigdm.py)
# ---------------------------------------------------------------------------

def _complex_randn(H: int, W: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    return (torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)) / np.sqrt(2)


def _psnr(x_gt: torch.Tensor, x_recon: torch.Tensor) -> float:
    gt_mag = x_gt.abs()
    recon_mag = x_recon.abs()
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
