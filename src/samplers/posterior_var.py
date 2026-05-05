"""Tweedie second-moment posterior variance estimation.

For a diffusion model with denoiser μ_θ, Tweedie's identity gives the
posterior mean E[x_0|x_t] = μ_θ(x_t). The second-moment identity
(Efron 2011, Stein 1981) gives the posterior covariance:

        Cov[x_0 | x_t]  =  σ_t^2 · ∂ μ_θ(x_t) / ∂ x_t           (J)

J is symmetric positive-semidefinite for a proper score network. We need
the *diagonal of its k-space conjugate* to drive frequency-adaptive data
consistency (the "Posterior-Variance-Gated DC" of FA-KGD-PV):

        P_i(t) = diag( F · σ_t^2 J · F^H )_i

Computing P_i exactly costs N denoiser calls per step (one per basis
vector). We use a single-probe **Hutchinson estimator** that costs one
extra denoiser call:

        v ~ CN(0, I)                            (k-space white probe)
        v_img = F^H v
        Jv ≈ ( μ_θ(x_t + ε v_img) − μ_θ(x_t) ) / ε
        P̂_i  =  σ_t^2 · |F(Jv)|_i^2  / mean(|v|^2)              (★)

A second pass with v → −v gives a centered finite-difference and removes
the O(ε) bias for free. Optionally, we radially smooth P̂ to a low-degree
polynomial in r — exactly matching the FA-KGD radial parameterisation —
which makes the per-step cost negligibly noisy.
"""
from __future__ import annotations

import torch
from typing import Callable

from .mri_forward import fft2c, build_radius_grid


def _complex_white(H: int, W: int, device, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        re = torch.randn(H, W, device=device)
        im = torch.randn(H, W, device=device)
    else:
        re = torch.randn(H, W, generator=generator).to(device)
        im = torch.randn(H, W, generator=generator).to(device)
    return (re + 1j * im) / (2 ** 0.5)


@torch.no_grad()
def estimate_posterior_variance_kspace(
    x_t: torch.Tensor,
    mu: torch.Tensor,
    sigma_t: float,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    eps_probe: float = 1e-2,
    centered: bool = True,
    radial_smooth: bool = True,
    radial_degree: int = 4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hutchinson estimate of the per-frequency posterior variance P̂_i(t).

    Args:
        x_t: current diffusion iterate (H,W) complex
        mu : already-computed denoiser output μ_θ(x_t)
        sigma_t: current noise level σ_t
        denoiser_fn: callable(x, σ) -> denoised image
        eps_probe: finite-difference step (relative to σ_t·||v||)
        centered: if True, do a centred FD (2 NFEs); else 1 NFE
        radial_smooth: if True, fit a low-degree radial polynomial
        radial_degree: degree of even polynomial in (r/r_max)

    Returns:
        P̂  : (H,W) real, per-frequency posterior variance estimate
        Jv  : (H,W) complex JVP (image-domain) — useful for diagnostics
    """
    H, W = x_t.shape[-2], x_t.shape[-1]
    device = x_t.device

    # Probe vector — k-space white, transformed to image
    v_kspace = _complex_white(H, W, device, generator=generator)
    v_img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(v_kspace),
                                                 norm="ortho"))
    # Renormalise so ||v_img||² ≈ HW (roughly unit-RMS perturbation)
    v_norm = v_img.abs().pow(2).mean().sqrt().clamp(min=1e-12)
    v_img = v_img / v_norm
    v_kspace = v_kspace / v_norm  # keep them paired

    # Choose ε so the perturbation is small relative to current iterate
    eps = eps_probe * sigma_t

    if centered:
        mu_p = denoiser_fn(x_t + eps * v_img, sigma_t)
        mu_m = denoiser_fn(x_t - eps * v_img, sigma_t)
        Jv = (mu_p - mu_m) / (2.0 * eps)
    else:
        mu_p = denoiser_fn(x_t + eps * v_img, sigma_t)
        Jv = (mu_p - mu) / eps

    # Push to k-space, square magnitude — gives stochastic diag estimate
    Jv_k = fft2c(Jv)
    P = (Jv_k.abs() ** 2) * (sigma_t ** 2)
    # Normalise by probe energy (already ≈1, but safe)
    P = P / v_kspace.abs().pow(2).mean().clamp(min=1e-12)

    if radial_smooth:
        P = _radial_polyfit(P, degree=radial_degree)

    # Floor to a small positive value
    P = P.clamp(min=1e-12)
    return P, Jv


def _radial_polyfit(P: torch.Tensor, degree: int = 4) -> torch.Tensor:
    """Fit a low-degree even polynomial in r to a 2D positive map.

    Returns a smooth radially-symmetric (H,W) map. Uses a closed-form
    weighted least squares fit in log-domain to keep the result positive
    and well-conditioned.
    """
    H, W = P.shape[-2:]
    device = P.device
    r = build_radius_grid(H, W).to(device)
    r_max = r.max().clamp(min=1.0)
    rn = (r / r_max).flatten()
    y = P.flatten().clamp(min=1e-20).log()  # log-fit to stay positive
    # Even polynomial basis in rn: 1, rn^2, rn^4, ...
    cols = [torch.ones_like(rn)]
    for k in range(1, degree + 1):
        cols.append(rn ** (2 * k))
    Phi = torch.stack(cols, dim=1)  # (N, degree+1)
    # Solve  Phi β ≈ y  in the least-squares sense
    sol = torch.linalg.lstsq(Phi, y.unsqueeze(1)).solution.squeeze(1)
    y_smooth = (Phi @ sol).exp()
    return y_smooth.view(H, W)
