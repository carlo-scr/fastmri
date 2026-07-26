"""Tweedie second-moment predictive-variance estimation.

For a diffusion model with denoiser μ_θ, Tweedie's identity gives the
posterior mean E[x_0|x_t] = μ_θ(x_t). The second-moment identity
(Efron 2011, Stein 1981) gives the posterior covariance:

        Cov[x_0 | x_t]  =  σ_t^2 · ∂ μ_θ(x_t) / ∂ x_t           (J)

where the equality is in the real coordinate representation used by the
network. For a proper score, J is symmetric positive semidefinite. We need
the diagonal of the covariance projected into a candidate measurement basis:

        P_i(t) = diag( F · σ_t^2 J · F^H )_i

Computing P exactly costs one denoiser call per real measurement coordinate.
We instead use a cross-Hutchinson estimator. In stacked real/imaginary
coordinates, draw a Rademacher probe z with unit variance per component:

        u = F^H z
        Ju ≈ [μ_θ(x_t + εu) − μ_θ(x_t - εu)] / (2ε)
        P̂_i = σ_t^2 Re(conj(z_i) [FJu]_i).                       (★)

The expectation of (★) is the sum of the real and imaginary marginal
variances for each complex k-space coordinate, directly comparable with a
complex measurement-noise variance E|n_i|². This cross product is essential:
the previously used |FJu|² estimates diag(F J J^T F^T), i.e. squared
Jacobian row energy, and is not a posterior variance.

This module handles the single-coil Fourier basis. Multi-coil acquisition
must probe the full SENSE candidate operator; see ``operator_sketch.py``.
"""
from __future__ import annotations

import torch
from typing import Callable

from .mri_forward import fft2c, ifft2c, build_radius_grid


def _complex_rademacher(
    H: int,
    W: int,
    device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Complex probe whose real and imaginary coordinates each have variance 1."""
    shape = (H, W)
    if generator is None:
        re = torch.empty(shape, device=device).bernoulli_(0.5).mul_(2).sub_(1)
        im = torch.empty(shape, device=device).bernoulli_(0.5).mul_(2).sub_(1)
    else:
        re = torch.randint(0, 2, shape, generator=generator).to(device=device)
        im = torch.randint(0, 2, shape, generator=generator).to(device=device)
        re = re.mul(2).sub(1)
        im = im.mul(2).sub(1)
    return re + 1j * im


def _real_coordinate_rms(x: torch.Tensor) -> torch.Tensor:
    """RMS over the stacked real representation of a real or complex tensor."""
    if torch.is_complex(x):
        return ((x.real.square().mean() + x.imag.square().mean()) / 2).sqrt()
    return x.square().mean().sqrt()


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
    n_probes: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate complex-coordinate predictive variance in the Fourier basis.

    Args:
        x_t: current diffusion iterate (H,W) complex
        mu : already-computed denoiser output μ_θ(x_t)
        sigma_t: current noise level σ_t
        denoiser_fn: callable(x, σ) -> denoised image
        eps_probe: central-FD step as a fraction of σ_t after normalizing the
            pullback direction to unit RMS in stacked real coordinates
        centered: if True, do a centred FD (2 NFEs); else 1 NFE
        radial_smooth: if True, fit a low-degree radial polynomial
        radial_degree: degree of even polynomial in (r/r_max)

    Returns:
        P̂  : (H,W) real, per-frequency posterior variance estimate
        Jv  : (H,W) complex JVP (image-domain) — useful for diagnostics
    """
    H, W = x_t.shape[-2], x_t.shape[-1]
    device = x_t.device
    delta = max(float(eps_probe) * max(abs(float(sigma_t)), 1e-8), 1e-8)

    P_acc = torch.zeros((H, W), device=device, dtype=x_t.real.dtype)
    Jv_last = None
    for _ in range(max(1, n_probes)):
        # z has identity covariance in the stacked real/imag representation.
        z_kspace = _complex_rademacher(H, W, device, generator=generator)
        pullback = ifft2c(z_kspace)
        pullback_rms = _real_coordinate_rms(pullback).clamp(min=1e-12)
        direction = pullback / pullback_rms

        if centered:
            mu_p = denoiser_fn(x_t + delta * direction, sigma_t)
            mu_m = denoiser_fn(x_t - delta * direction, sigma_t)
            J_direction = (mu_p - mu_m) / (2.0 * delta)
        else:
            mu_p = denoiser_fn(x_t + delta * direction, sigma_t)
            J_direction = (mu_p - mu) / delta

        # Undo direction normalization so this is J F^H z.
        Jv = J_direction * pullback_rms
        Jv_k = fft2c(Jv)
        # Real-embedding Hutchinson cross product. Because Re(z) and Im(z)
        # each have unit variance, this estimates Var(Re k_i)+Var(Im k_i).
        P_one = (float(sigma_t) ** 2) * (
            z_kspace.real * Jv_k.real + z_kspace.imag * Jv_k.imag
        )
        P_acc = P_acc + P_one
        Jv_last = Jv

    P = P_acc / max(1, n_probes)

    if radial_smooth:
        # The exact diagonal is nonnegative, but a finite-probe Hutchinson
        # estimate need not be. Clip only after probe averaging.
        P = _radial_polyfit(P.clamp(min=1e-12), degree=radial_degree)

    P = P.clamp(min=0)
    return P, Jv_last


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
