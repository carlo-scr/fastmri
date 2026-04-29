"""ACS-based noise variance estimation for FA-KGD.

Two estimators are exposed:

* `estimate_sigma_sq_per_slice` — per-frequency sample variance over the
  remaining (e.g. coil or replicate) axis of one slice's ACS box.
* `estimate_sigma_sq_pooled` — same estimator pooled over the slice axis of
  a whole volume. Effective d.o.f. multiply by `N_z`, so MSE drops by
  ~`N_z` for stationary noise.

Both return a real `(H, W)` map covering the full k-space grid, with values
outside the ACS region filled by a smooth radial polynomial fit.
"""
from __future__ import annotations

import numpy as np
import torch


def _acs_box(H: int, W: int, center_fraction: float) -> tuple[int, int, int, int]:
    acs_h = max(2, int(round(center_fraction * H)))
    acs_w = max(2, int(round(center_fraction * W)))
    sy = H // 2 - acs_h // 2
    sx = W // 2 - acs_w // 2
    return sy, sx, acs_h, acs_w


def _radial_extrapolate(
    pooled_acs: torch.Tensor, H: int, W: int, sy: int, sx: int,
) -> torch.Tensor:
    """Quadratic radial fit through ACS samples; replace ACS region with the
    measured values, polynomial elsewhere."""
    acs_h, acs_w = pooled_acs.shape
    full = torch.full((H, W), float("nan"), dtype=torch.float32)
    full[sy : sy + acs_h, sx : sx + acs_w] = pooled_acs.to(torch.float32)

    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((gy - cy).float() ** 2 + (gx - cx).float() ** 2)

    inside = ~torch.isnan(full)
    r_in = r[inside].numpy()
    v_in = full[inside].numpy()
    coefs = np.polyfit(r_in, v_in, deg=2)
    fit = torch.from_numpy(np.polyval(coefs, r.numpy())).to(torch.float32)
    fit = fit.clamp(min=float(v_in.min()) * 0.5)
    fit[inside] = full[inside]
    return fit


def estimate_sigma_sq_per_slice(
    y_slice: torch.Tensor,
    center_fraction: float = 0.08,
    fit_radial: bool = True,
) -> torch.Tensor:
    """Per-slice ACS noise variance.

    Args:
        y_slice: complex k-space of shape (..., H, W). The leading dim is the
            "replicate" axis (e.g. coils, or a singleton). Variance is taken
            over that axis with Bessel correction.
        center_fraction: ACS fraction (matches the rest of the pipeline).
        fit_radial: If True, extrapolate outside the ACS via quadratic fit.

    Returns:
        Real `(H, W)` variance map.
    """
    if y_slice.dim() < 2:
        raise ValueError("y_slice must be at least 2D (..., H, W)")
    if y_slice.dim() == 2:
        y_slice = y_slice.unsqueeze(0)  # singleton replicate
    H, W = y_slice.shape[-2:]
    sy, sx, ah, aw = _acs_box(H, W, center_fraction)
    acs = y_slice[..., sy : sy + ah, sx : sx + aw]  # (..., ah, aw)
    n_rep = int(np.prod(acs.shape[:-2]))
    if n_rep < 2:
        raise ValueError(
            "Per-slice ACS variance estimation needs ≥2 replicates along the "
            "leading axis (e.g. coils). With single-coil / RSS-combined data "
            "this estimator is degenerate; use estimate_sigma_sq_pooled across "
            "slices instead."
        )
    flat = acs.reshape(n_rep, ah, aw)
    mean_r = flat.mean(dim=0, keepdim=True)
    pooled = ((flat - mean_r).abs() ** 2).sum(dim=0) / (n_rep - 1)
    pooled = pooled.cpu()
    if not fit_radial:
        full = torch.full((H, W), float(pooled.mean()), dtype=torch.float32)
        full[sy : sy + ah, sx : sx + aw] = pooled.to(torch.float32)
        return full
    return _radial_extrapolate(pooled, H, W, sy, sx)


def estimate_sigma_sq_pooled(
    y_volume: torch.Tensor,
    center_fraction: float = 0.08,
    fit_radial: bool = True,
) -> torch.Tensor:
    """Volumetric (slice-pooled) ACS noise variance.

    Args:
        y_volume: complex k-space of shape (Nz, ..., H, W). All leading dims
            other than the H, W axes are pooled (e.g. (Nz, Nc, H, W) pools
            both slices and coils, (Nz, H, W) pools only slices).
    """
    if y_volume.dim() < 2:
        raise ValueError("y_volume must be at least (Nz, H, W)")
    H, W = y_volume.shape[-2:]
    sy, sx, ah, aw = _acs_box(H, W, center_fraction)
    acs = y_volume[..., sy : sy + ah, sx : sx + aw]
    # Flatten everything except (H, W) into a single replicate axis
    n_rep = int(np.prod(acs.shape[:-2]))
    if n_rep < 2:
        raise ValueError("Pooled estimator needs ≥2 samples along leading dims")
    flat = acs.reshape(n_rep, ah, aw)
    mean_r = flat.mean(dim=0, keepdim=True)
    pooled = ((flat - mean_r).abs() ** 2).sum(dim=0) / (n_rep - 1)
    pooled = pooled.cpu()
    if not fit_radial:
        full = torch.full((H, W), float(pooled.mean()), dtype=torch.float32)
        full[sy : sy + ah, sx : sx + aw] = pooled.to(torch.float32)
        return full
    return _radial_extrapolate(pooled, H, W, sy, sx)


__all__ = ["estimate_sigma_sq_per_slice", "estimate_sigma_sq_pooled"]
