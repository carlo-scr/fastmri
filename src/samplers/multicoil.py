"""Multi-coil SENSE versions of ΠGDM and FA-KGD+FPDC.

These mirror `pigdm.py` and `fakgd.py` but operate on multi-coil k-space
through a `MultiCoilSENSE` operator. The denoiser still consumes a single
complex-valued image (the SENSE-combined estimate); per-coil information
enters only through the data-consistency / Kalman update.

The Kalman gain becomes per-coil-per-frequency:

    K_{c,i}(t) = σ_t^2 / (σ_t^2 + σ_{c,i}^2)

For ΠGDM we use the per-coil scalar variance directly. For FA-KGD we
broadcast a per-frequency `σ_i^2(r)` map across coils, optionally scaled
by the per-coil scalar so we model both sources of heteroscedasticity.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c, build_radius_grid
from .sense import MultiCoilSENSE
from .fakgd import fpdc_radius, fpdc_gate


def _complex_randn(H: int, W: int, device=torch.device("cpu")) -> torch.Tensor:
    return (torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)) / np.sqrt(2)


def _psnr(x_gt: torch.Tensor, x_recon: torch.Tensor) -> float:
    gt_mag = x_gt.abs()
    recon_mag = x_recon.abs()
    H, W = min(gt_mag.shape[-2], 320), min(gt_mag.shape[-1], 320)
    sh, sw = (gt_mag.shape[-2] - H) // 2, (gt_mag.shape[-1] - W) // 2
    gt_mag = gt_mag[..., sh : sh + H, sw : sw + W]
    recon_mag = recon_mag[..., sh : sh + H, sw : sw + W]
    data_range = gt_mag.max()
    mse = ((gt_mag - recon_mag) ** 2).mean()
    if mse < 1e-12:
        return 100.0
    return (10 * torch.log10(data_range**2 / mse)).item()


# ---------------------------------------------------------------------------
# ΠGDM, multi-coil SENSE
# ---------------------------------------------------------------------------

def run_pigdm_mc(
    y_mc: torch.Tensor,
    sense_op: MultiCoilSENSE,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_y: Optional[float] = None,
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> dict:
    """Multi-coil ΠGDM with isotropic per-coil noise scalar.

    Args:
        y_mc: Multi-coil k-space (Nc,H,W).
        sense_op: MultiCoilSENSE operator (mask + sens + per-coil σ_c^2).
        sigma_schedule: descending noise schedule (T,).
        denoiser_fn: callable(complex [H,W], σ) -> complex [H,W].
        sigma_y: if given, override per-coil σ_c^2 with a scalar (true ΠGDM
            isotropic baseline). If None we use sense_op.sigma_c_sq directly.
    """
    torch.manual_seed(seed)
    Nc, H, W = y_mc.shape
    device = y_mc.device
    T = len(sigma_schedule)

    # Per-coil variance (scalar per coil, broadcast to (Nc,1,1))
    if sigma_y is not None:
        sig2 = torch.full((Nc,), float(sigma_y) ** 2, device=device)
    else:
        sig2 = sense_op.sigma_c_sq.to(device)
    sig2_view = sig2.view(Nc, 1, 1)

    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)
    psnr_traj = []

    for step in range(T):
        sigma_t = sigma_schedule[step].item()
        mu = denoiser_fn(x_t, sigma_t)              # (H,W) complex

        # Per-coil k-space residual
        mu_kc = sense_op.forward(mu)                # (Nc,H,W)
        residual = sense_op.mask * (y_mc - mu_kc)   # (Nc,H,W)

        # Per-coil Kalman gain (scalar per coil)
        K = sigma_t ** 2 / (sigma_t ** 2 + sig2_view)  # (Nc,1,1)

        # Combine corrections with SENSE adjoint (Roemer-weighted)
        corr = sense_op.adjoint(K * residual)       # (H,W)
        x_corrected = mu + corr

        if step < T - 1:
            sigma_next = sigma_schedule[step + 1].item()
            x_t = x_corrected + sigma_next * _complex_randn(H, W, device=device)
        else:
            x_t = x_corrected

        if x_gt is not None:
            psnr_traj.append(_psnr(x_gt, x_t))

    return {"recon": x_t, "psnr_trajectory": psnr_traj}


# ---------------------------------------------------------------------------
# FA-KGD + FPDC, multi-coil SENSE
# ---------------------------------------------------------------------------

def run_fakgd_mc(
    y_mc: torch.Tensor,
    sense_op: MultiCoilSENSE,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_i_sq_init: torch.Tensor,
    r_acs: float,
    r_max: float,
    beta_fpdc: float = 1.0,
    alpha_ema: float = 0.95,
    gamma: float = 0.0,
    eps: float = 1e-8,
    m_step_mode: str = "clamp",
    m_step_start_frac: float = 0.0,
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> dict:
    """Multi-coil FA-KGD + FPDC.

    Args:
        y_mc: Multi-coil k-space (Nc,H,W).
        sense_op: MultiCoilSENSE operator (mask, sens, σ_c^2).
        sigma_i_sq_init: (H,W) per-frequency variance map (shared across
            coils). The effective per-(coil,freq) variance is
                σ_{c,i}^2 = σ_i^2(r) · (σ_c^2 / mean(σ_c^2))
            i.e. the per-coil scalar modulates the per-frequency profile.
        Other args identical to single-coil run_fakgd.
    """
    torch.manual_seed(seed)
    Nc, H, W = y_mc.shape
    device = y_mc.device
    T = len(sigma_schedule)

    radius_grid = build_radius_grid(H, W).to(device)

    # Per-coil scaling (broadcastable to (Nc,1,1))
    sig_c = sense_op.sigma_c_sq.to(device)
    sig_c_norm = sig_c / sig_c.mean().clamp(min=eps)
    sig_c_view = sig_c_norm.view(Nc, 1, 1)

    sigma_i_sq = sigma_i_sq_init.clone().to(device)  # (H,W) per-freq map

    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)
    psnr_traj = []
    gain_maps = {}

    for step in range(T):
        sigma_t = sigma_schedule[step].item()
        mu = denoiser_fn(x_t, sigma_t)              # (H,W)

        # FPDC: progressive frequency mask
        r_t = fpdc_radius(step, T, r_acs, r_max, beta_fpdc)
        freq_gate = fpdc_gate(radius_grid, r_t)
        mask_t = sense_op.mask * freq_gate          # (H,W)

        # Per-coil k-space residual
        mu_kc = fft2c(sense_op.sens * mu.unsqueeze(0))  # (Nc,H,W)
        residual = mask_t * (y_mc - mu_kc)           # (Nc,H,W)

        alpha_t = 1.0 - (1.0 - alpha_ema) * (sigma_t / sigma_max)
        progress = step / max(T - 1, 1)
        m_step_active = m_step_mode != "off" and progress >= m_step_start_frac
        if m_step_active:
            # Aggregate per-frequency innovation across coils.
            # Coil-c residual energy ≈ |S_c|^2 · σ_i^2(r) + (denoiser term).
            # Sum over coils, divide by sum |S_c|^2 to recover per-freq σ^2.
            num = (residual.abs() ** 2).sum(dim=0)
            den = (sense_op.sens.abs() ** 2).sum(dim=0).clamp(min=eps)
            innov_per_freq = num / den
            innov_per_freq = torch.clamp(innov_per_freq - gamma * sigma_t**2, min=eps)
            update_gate = (mask_t > 0)
            sigma_i_sq_new = alpha_t * sigma_i_sq + (1 - alpha_t) * innov_per_freq
            if m_step_mode == "clamp":
                sigma_i_sq_new = torch.minimum(sigma_i_sq_new, sigma_i_sq_init)
            sigma_i_sq = torch.where(update_gate, sigma_i_sq_new, sigma_i_sq)

        # Per-coil per-freq variance and Kalman gain
        # σ_{c,i}^2 = σ_i^2(r) · sig_c_norm[c]
        sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view   # (Nc,H,W)
        K = sigma_t**2 / (sigma_t**2 + sig_ci_sq)          # (Nc,H,W)

        corr = sense_op.adjoint(K * residual)               # (H,W)
        x_corrected = mu + corr

        if step < T - 1:
            sigma_next = sigma_schedule[step + 1].item()
            x_t = x_corrected + sigma_next * _complex_randn(H, W, device=device)
        else:
            x_t = x_corrected

        if x_gt is not None:
            psnr_traj.append(_psnr(x_gt, x_t))
        if return_diagnostics and step in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
            gain_maps[step] = K.detach().mean(dim=0).cpu().numpy()

    out = {"recon": x_t, "psnr_trajectory": psnr_traj,
           "sigma_i_sq_final": sigma_i_sq}
    if return_diagnostics:
        out["gain_maps"] = gain_maps
    return out
