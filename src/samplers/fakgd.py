"""FA-KGD + FPDC sampler: Frequency-Adaptive Kalman-Guided Diffusion
with Frequency-Progressive Data Consistency.

Implements Algorithm 1 from the proposal (v7).
Three synergistic components:
  1. ACS-calibrated frequency-adaptive Kalman gain K_i(t)
  2. FPDC: r(t) = r_ACS + (r_max - r_ACS)(1 - t/T)^β
  3. γ-corrected M-step with time-decaying α  (γ=0 recommended)
"""

import torch
import numpy as np
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c, build_radius_grid


# ---------------------------------------------------------------------------
# FPDC helpers
# ---------------------------------------------------------------------------

def fpdc_radius(step: int, T: int, r_acs: float, r_max: float, beta: float) -> float:
    """Compute the FPDC frequency radius at denoising step `step`.

    step=0 is the start (high noise) → r = r_acs (ACS only).
    step=T-1 is the end (low noise) → r = r_max (full k-space).
    """
    if T <= 1:
        return r_max
    progress = step / (T - 1)  # 0 → 1 as denoising proceeds
    return r_acs + (r_max - r_acs) * progress ** beta


def fpdc_gate(radius_grid: torch.Tensor, r_t: float) -> torch.Tensor:
    """Binary gate: 1 inside radius r_t, 0 outside."""
    return (radius_grid <= r_t).float()


# ---------------------------------------------------------------------------
# Main sampler
# ---------------------------------------------------------------------------

def run_fakgd(
    y: torch.Tensor,
    mask: torch.Tensor,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_i_sq_init: torch.Tensor,
    # FPDC parameters
    r_acs: float,
    r_max: float,
    beta_fpdc: float = 1.0,
    # EM parameters
    alpha_ema: float = 0.95,
    gamma: float = 0.0,
    eps: float = 1e-8,
    m_step_mode: str = "full",  # "full", "clamp", "off"
    m_step_start_frac: float = 0.0,  # only run M-step once denoising progress >= frac
    # Optional
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> dict:
    """Run FA-KGD + FPDC reconstruction.

    Args:
        y: Measured k-space (complex, [H, W]).
        mask: Undersampling mask (real, broadcastable to [H, W]).
        sigma_schedule: Noise levels descending from σ_max to σ_min, shape (T,).
        denoiser_fn: Callable(x_t, sigma_t) -> mu_theta.
        sigma_i_sq_init: Initial per-frequency noise variance estimate (real, [H, W]).
            Typically from ACS calibration.
        r_acs: ACS frequency radius (inner boundary of FPDC).
        r_max: Maximum frequency radius (full k-space extent).
        beta_fpdc: FPDC schedule exponent (default 1.0).
        alpha_ema: Base EMA smoothing for M-step (default 0.95).
        gamma: Bias correction factor (0 = no correction, recommended).
        eps: Floor for clamping innovation (numerical stability).
        m_step_mode: M-step update mode:
            "full": standard EMA update (works with oracle).
            "clamp": EMA update clamped to never exceed σ_i_sq_init
                (prevents explosion from denoiser error at high σ_t).
            "off": no M-step, use fixed σ_i_sq_init throughout.
        m_step_start_frac: only run the M-step once denoising progress
            reaches this fraction of the total steps. Default 0.0 keeps the
            original behaviour (run from step 0). Setting e.g. 0.5 freezes
            σ_i² during the first half of sampling, where the residual is
            dominated by denoiser error rather than measurement noise; this
            lets `m_step_mode="full"` actually grow σ_i² at high frequencies
            (where true noise is larger) without the denoiser-error blow-up
            that motivates `clamp`. Use together with
            `noise_init=multicoil_acs` (flat init) to recover frequency
            structure from the residuals.
        x_gt: Ground truth for PSNR tracking (optional).
        seed: Random seed.
        return_diagnostics: If True, return gain maps and sigma trajectories.

    Returns:
        dict with keys:
            'recon': complex [H, W] reconstruction
            'psnr_trajectory': list of PSNR values per step
            'sigma_i_sq_final': final noise variance map
            (if return_diagnostics):
            'gain_maps': dict of step -> K map
            'sigma_trajectory': array of [low, mid, high] band sigmas per step
    """
    torch.manual_seed(seed)
    T = len(sigma_schedule)
    H, W = y.shape[-2], y.shape[-1]
    device = y.device

    # Build frequency radius grid
    radius_grid = build_radius_grid(H, W).to(device)

    # Initialize
    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)
    sigma_i_sq = sigma_i_sq_init.clone().to(device)

    psnr_trajectory = []
    sigma_trajectory = []
    gain_maps = {}

    for step in range(T):
        sigma_t = sigma_schedule[step].item()

        # --- Denoiser prediction ---
        mu_theta = denoiser_fn(x_t, sigma_t)

        # --- FPDC: progressive frequency gating ---
        r_t = fpdc_radius(step, T, r_acs, r_max, beta_fpdc)
        freq_gate = fpdc_gate(radius_grid, r_t)
        mask_t = mask * freq_gate

        # --- K-space residual ---
        mu_k = fft2c(mu_theta)
        residual = mask_t * (y - mu_k)

        # --- Time-decaying α (Eq. 8) ---
        alpha_t = 1.0 - (1.0 - alpha_ema) * (sigma_t / sigma_max)

        # --- M-step: online noise variance update ---
        # Gate by denoising progress rather than raw sigma values. EDM sigma
        # schedules are highly non-linear, so a threshold like sigma_t <= 0.5
        # * sigma_max would activate far too early (e.g. step 3/20 for the
        # default EDM schedule). start_frac=0.5 should mean "start halfway
        # through the denoising steps".
        progress = step / max(T - 1, 1)
        m_step_active = m_step_mode != "off" and progress >= m_step_start_frac
        if m_step_active:
            residual_sq = residual.abs() ** 2
            innovation = torch.clamp(residual_sq - gamma * sigma_t**2, min=eps)
            update_mask = mask_t > 0
            sigma_i_sq_new = alpha_t * sigma_i_sq + (1 - alpha_t) * innovation
            if m_step_mode == "clamp":
                # Prevent σ_i² from exceeding initial estimate (guards against denoiser error)
                sigma_i_sq_new = torch.minimum(sigma_i_sq_new, sigma_i_sq_init)
            sigma_i_sq = torch.where(update_mask, sigma_i_sq_new, sigma_i_sq)

        # --- Frequency-adaptive Kalman gain ---
        K = sigma_t**2 / (sigma_t**2 + sigma_i_sq)

        # --- Kalman-corrected image ---
        x_corrected = mu_theta + ifft2c(K * residual)

        # --- Reverse step ---
        if step < T - 1:
            sigma_next = sigma_schedule[step + 1].item()
            x_t = x_corrected + sigma_next * _complex_randn(H, W, device=device)
        else:
            x_t = x_corrected

        # --- Tracking ---
        if x_gt is not None:
            p = _psnr(x_gt, x_t)
            psnr_trajectory.append(p)

        if return_diagnostics:
            if step in [0, T // 4, T // 2, 3 * T // 4, T - 1]:
                gain_maps[step] = K.detach().cpu().numpy()
            r_int = radius_grid.int()
            r_mid = int(r_max * 0.3)
            r_high = int(r_max * 0.7)
            sigma_trajectory.append([
                sigma_i_sq[r_int == 5].mean().item() if (r_int == 5).any() else 0,
                sigma_i_sq[r_int == r_mid].mean().item() if (r_int == r_mid).any() else 0,
                sigma_i_sq[r_int == r_high].mean().item() if (r_int == r_high).any() else 0,
            ])

    result = {
        "recon": x_t,
        "psnr_trajectory": psnr_trajectory,
        "sigma_i_sq_final": sigma_i_sq,
    }
    if return_diagnostics:
        result["gain_maps"] = gain_maps
        result["sigma_trajectory"] = np.array(sigma_trajectory)
    return result


class FAKGDSampler:
    """Object-oriented wrapper around run_fakgd."""

    def __init__(
        self,
        denoiser_fn: Callable,
        alpha_ema: float = 0.95,
        gamma: float = 0.0,
        beta_fpdc: float = 1.0,
        eps: float = 1e-8,
        m_step_mode: str = "full",
        m_step_start_frac: float = 0.0,
    ):
        self.denoiser_fn = denoiser_fn
        self.alpha_ema = alpha_ema
        self.gamma = gamma
        self.beta_fpdc = beta_fpdc
        self.eps = eps
        self.m_step_mode = m_step_mode
        self.m_step_start_frac = m_step_start_frac

    def reconstruct(
        self,
        y: torch.Tensor,
        mask: torch.Tensor,
        sigma_schedule: torch.Tensor,
        sigma_i_sq_init: torch.Tensor,
        r_acs: float,
        r_max: float,
        x_gt: Optional[torch.Tensor] = None,
        seed: int = 0,
        return_diagnostics: bool = False,
    ) -> dict:
        return run_fakgd(
            y=y,
            mask=mask,
            sigma_schedule=sigma_schedule,
            denoiser_fn=self.denoiser_fn,
            sigma_i_sq_init=sigma_i_sq_init,
            r_acs=r_acs,
            r_max=r_max,
            beta_fpdc=self.beta_fpdc,
            alpha_ema=self.alpha_ema,
            gamma=self.gamma,
            eps=self.eps,
            m_step_mode=self.m_step_mode,
            m_step_start_frac=self.m_step_start_frac,
            x_gt=x_gt,
            seed=seed,
            return_diagnostics=return_diagnostics,
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
