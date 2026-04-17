"""Noise level schedules for diffusion sampling."""

import torch
import numpy as np


def ddpm_sigma_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """DDPM linear beta schedule → sigma values (descending: sigma_T … sigma_1).

    Uses σ_t = sqrt(1 - ᾱ_t), matching the VP noise level convention
    where x_t = sqrt(ᾱ_t)·x_0 + σ_t·ε.

    Returns tensor of shape (T,) with sigma_t for t = T, T-1, …, 1.
    """
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sigmas = (1 - alpha_bar).sqrt()
    return sigmas.flip(0).float()  # descending: largest → smallest


def edm_sigma_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> torch.Tensor:
    """EDM geometric sigma schedule (Karras et al. 2022).

    Returns tensor of shape (num_steps,) with sigma values descending
    from sigma_max to sigma_min.
    """
    step_indices = torch.arange(num_steps, dtype=torch.float64)
    sigmas = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return sigmas.float()
