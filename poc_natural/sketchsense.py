"""Batched natural-image instantiation of operator-space SketchSense."""
from __future__ import annotations

import math

import torch

from poc_natural.pvas_core import EpsDenoiser
from src.samplers.operator_sketch import (
    OperatorCovarianceSketch,
    operator_sketch_from_products,
    real_flatten,
)


@torch.no_grad()
def sketch_identity_measurements_batched(
    denoiser: EpsDenoiser,
    xt: torch.Tensor,
    t: int,
    *,
    num_probes: int,
    fd_step: float,
    generator: torch.Generator,
    psd_rtol: float = 1e-6,
) -> list[OperatorCovarianceSketch]:
    """Sketch per-image pixel-space covariance with batched denoiser calls.

    The candidate operator is the identity over all image channels/pixels.
    Although JVP calls are batched, each image receives an independent probe
    matrix and an independently factorized covariance sketch.
    """
    if num_probes <= 0:
        raise ValueError("num_probes must be positive")
    if fd_step <= 0:
        raise ValueError("fd_step must be positive")
    batch = xt.shape[0]
    a = denoiser.abar[t].to(device=xt.device, dtype=xt.dtype)
    sigma_sq = ((1.0 - a) / a).item()
    perturb_scale = float(fd_step) * math.sqrt(float(a.item()))

    probe_shape = (num_probes,) + tuple(xt.shape)
    probes = torch.randint(
        0, 2, probe_shape, generator=generator, device="cpu"
    ).to(device=xt.device, dtype=xt.dtype).mul_(2).sub_(1)
    responses = torch.empty_like(probes)
    for probe_index in range(num_probes):
        direction = probes[probe_index]
        mu_plus = denoiser.x0hat(
            xt + perturb_scale * direction, t
        )
        mu_minus = denoiser.x0hat(
            xt - perturb_scale * direction, t
        )
        jvp = (mu_plus - mu_minus) / (2.0 * float(fd_step))
        responses[probe_index] = sigma_sq * jvp

    sketches: list[OperatorCovarianceSketch] = []
    for image_index in range(batch):
        omega = torch.stack([
            real_flatten(probes[q, image_index])
            for q in range(num_probes)
        ], dim=1)
        response = torch.stack([
            real_flatten(responses[q, image_index])
            for q in range(num_probes)
        ], dim=1)
        template = torch.zeros_like(xt[image_index])
        sketches.append(operator_sketch_from_products(
            measurement_template=template,
            image_template=template,
            omega=omega,
            response=response,
            image_products=response,
            psd_rtol=psd_rtol,
            diagnostics={
                "vp_timestep": int(t),
                "zero_pullbacks": 0,
            },
        ))
    return sketches


def block_atom_ids(
    channels: int,
    height: int,
    width: int,
    block: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map every channel/pixel coordinate to its spatial block atom."""
    if height % block or width % block:
        raise ValueError("height and width must be divisible by block")
    grid_h, grid_w = height // block, width // block
    ids_small = torch.arange(
        grid_h * grid_w, device=device, dtype=torch.long
    ).reshape(grid_h, grid_w)
    ids = ids_small.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return ids.unsqueeze(0).expand(channels, height, width)


def average_pool_conditioning_gram(
    sketch: OperatorCovarianceSketch,
    block: int,
    noise_variance: float,
) -> torch.Tensor:
    """Low-rank precision contribution of an observed box-averaged image.

    The natural-image SR protocol always observes ``avg_pool2d(x, block)``.
    This operator is not one of the full-resolution candidate blocks, so its
    contribution is supplied as an explicit initial Gram matrix rather than
    pretending it is a candidate atom.
    """
    rank = sketch.rank
    if rank == 0:
        return sketch.measurement_factor.new_zeros((0, 0))
    template = sketch.measurement_template
    if template.dim() != 3 or torch.is_complex(template):
        raise ValueError(
            "average-pool conditioning expects a real (C,H,W) template"
        )
    if template.shape[-2] % block or template.shape[-1] % block:
        raise ValueError("measurement shape must be divisible by block")
    factors = sketch.measurement_factor.T.reshape(
        rank, *template.shape
    )
    pooled = torch.nn.functional.avg_pool2d(factors, block)
    rows = pooled.reshape(rank, -1).T
    return rows.T @ rows / max(float(noise_variance), 1e-12)
