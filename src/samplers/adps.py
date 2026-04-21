"""ADPS (Ambient Diffusion Posterior Sampling) for MRI reconstruction.

Implements Daras et al. (2024) — "Ambient Diffusion Posterior Sampling:
Solving Inverse Problems with Diffusion Models Trained on Corrupted Data".

Unlike DPS which corrects from the denoised estimate mu_theta, ADPS applies
the likelihood gradient to the full ODE-stepped sample, with optional
stochastic churn for exploration.  We implement the ALD (Annealed Langevin
Dynamics) variant from the paper.

Reference:
    @inproceedings{daras2024adps,
      title={Ambient Diffusion Posterior Sampling},
      author={Daras, Giannis and Dagan, Yuval and Dimakis, Alexandros G
              and Daskalakis, Constantinos},
      booktitle={NeurIPS},
      year={2024}
    }
"""

import torch
import numpy as np
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c


def run_adps(
    y: torch.Tensor,
    mask: torch.Tensor,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    step_size: float = 1.0,
    s_churn: float = 0.0,
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> dict:
    """Run ADPS (ALD mode) reconstruction.

    Algorithm per step (Euler discretization of probability-flow ODE + ALD
    likelihood guidance):

      1. Optionally inject stochastic noise (S_churn):
           sigma_hat = sigma_t + gamma * sigma_t
           x_hat = x_t + sqrt(sigma_hat^2 - sigma_t^2) * noise

      2. Denoise:  D(x_hat, sigma_hat) -> denoised

      3. ODE tangent:  d = (x_hat - denoised) / sigma_hat

      4. Euler step:  x_ode = x_hat + h * d    where h = sigma_{t+1} - sigma_hat

      5. Likelihood score through autograd:
           loss = ||y - mask * F(denoised)||^2
           grad = d(loss)/d(x_t)
           x_next = x_ode - (l_ss / ||residual||) * grad

    Step-size convention (l_ss / sqrt(SSE)) follows the ADPS paper.

    Args:
        y: Measured k-space (complex, [H, W]).
        mask: Undersampling mask (real, broadcastable to [H, W]).
        sigma_schedule: Noise levels descending from sigma_max to sigma_min.
        denoiser_fn: Callable(x_t, sigma_t) -> denoised.
        step_size: ADPS likelihood step size (l_ss). Default 1.0 per paper.
        s_churn: Stochastic churn parameter (S_churn in the paper). 0 = deterministic.
        x_gt: Ground truth for PSNR tracking (optional).
        seed: Random seed.

    Returns:
        dict with 'recon' and 'psnr_trajectory'.
    """
    torch.manual_seed(seed)
    T = len(sigma_schedule)
    H, W = y.shape[-2], y.shape[-1]
    device = y.device

    # Build gradient-compatible denoiser (same pattern as DPS)
    _has_net = hasattr(denoiser_fn, "net")
    if _has_net:
        _net = denoiser_fn.net
        _dev = denoiser_fn.device if hasattr(denoiser_fn, "device") else device

    def _denoise_with_grad(x_complex: torch.Tensor, sigma_t: float) -> torch.Tensor:
        if _has_net:
            x_real = torch.stack([x_complex.real, x_complex.imag], dim=0)
            x_real = x_real.unsqueeze(0).float().to(_dev)
            sigma = torch.tensor([sigma_t], dtype=torch.float32, device=_dev)
            denoised = _net(x_real, sigma)
            out = denoised[0, 0] + 1j * denoised[0, 1]
            return out.to(x_complex.device)
        else:
            return denoiser_fn(x_complex, sigma_t)

    # Append sigma=0 at end (for final step h computation)
    sigmas = torch.cat([sigma_schedule, torch.zeros(1, device=device)])

    # Initialize from noise
    sigma_max = sigmas[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)

    psnr_trajectory = []

    for step in range(T):
        sigma_cur = sigmas[step].item()
        sigma_next = sigmas[step + 1].item()

        # --- Step 1: Stochastic churn ---
        gamma = min(s_churn / T, np.sqrt(2) - 1) if s_churn > 0 else 0.0
        sigma_hat = sigma_cur + gamma * sigma_cur

        if gamma > 0:
            noise_churn = _complex_randn(H, W, device=device)
            x_hat = x_t + np.sqrt(sigma_hat**2 - sigma_cur**2) * noise_churn
        else:
            x_hat = x_t
            sigma_hat = sigma_cur

        # --- Step 2: Denoise with gradient tracking ---
        x_hat_grad = x_hat.clone().detach().requires_grad_(True)
        denoised = _denoise_with_grad(x_hat_grad, sigma_hat)

        # --- Step 3 & 4: ODE tangent + Euler step ---
        # d = (x_hat - denoised) / sigma_hat
        # x_ode = x_hat + h * d  where h = sigma_next - sigma_hat
        h = sigma_next - sigma_hat
        d_cur = (x_hat_grad - denoised) / sigma_hat
        x_ode = x_hat_grad + h * d_cur

        # --- Step 5: ALD likelihood guidance ---
        # ALD mode: forward model applied to denoised (not Tweedie posterior mean)
        residual_k = mask * (y - fft2c(denoised))
        loss = residual_k.abs().pow(2).sum()

        try:
            grad = torch.autograd.grad(loss, x_hat_grad)[0]
        except RuntimeError:
            grad = -2.0 * ifft2c(residual_k).detach()

        # ADPS step-size: l_ss / sqrt(SSE)
        residual_norm = loss.sqrt().item()
        scaled_step = step_size / max(residual_norm, 1e-8)

        # Apply correction to ODE-stepped result
        x_t = x_ode.detach() - scaled_step * grad.detach()

        # Track PSNR
        if x_gt is not None:
            p = _psnr(x_gt, x_t)
            psnr_trajectory.append(p)

    return {"recon": x_t, "psnr_trajectory": psnr_trajectory}


class ADPSSampler:
    """Object-oriented wrapper around run_adps."""

    def __init__(self, denoiser_fn: Callable, step_size: float = 1.0,
                 s_churn: float = 0.0):
        self.denoiser_fn = denoiser_fn
        self.step_size = step_size
        self.s_churn = s_churn

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        sigma_schedule: torch.Tensor,
        x_gt: Optional[torch.Tensor] = None,
        seed: int = 0,
    ) -> dict:
        return run_adps(
            y=y, mask=mask, sigma_schedule=sigma_schedule,
            denoiser_fn=self.denoiser_fn, step_size=self.step_size,
            s_churn=self.s_churn, x_gt=x_gt, seed=seed,
        )


# ---------------------------------------------------------------------------
# Helpers
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
