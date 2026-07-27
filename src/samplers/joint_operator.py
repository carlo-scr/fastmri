"""Joint image/operator uncertainty utilities for RaoSense.

Most diffusion active-imaging methods treat the forward operator as known.
This module supports a deliberately low-dimensional nuisance state
``phi``--for example rigid motion, a coil-map basis, a phase ramp, an angular
LiDAR offset, or CT center-of-rotation error--without constructing a dense
joint covariance.

Given an image branch with aligned state/measurement factors and an
independent nuisance factor, :func:`augment_branch_with_nuisance` creates the
block-aligned joint branch.  The ordinary RaoSense sufficient projection and
Bayesian update can then condition image and nuisance coordinates together.

The target-information diagnostic quantifies a specific failure of
known-operator design.  For one local Gaussian it reports

* information about the complete joint state;
* information about selected target-state rows while marginalizing nuisance;
* target information conditional on the nuisance being known.

The last quantity is the optimistic score implicitly used by an acquisition
method that ignores calibration uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .operator_sketch import real_flatten
from .rao_belief import DiffusionBeliefBranch


@dataclass(frozen=True)
class TargetInformationDiagnostic:
    joint_information: float
    target_information: float
    known_nuisance_information: float
    known_operator_optimism: float
    target_rank: int
    nuisance_rank: int


def augment_branch_with_nuisance(
    branch: DiffusionBeliefBranch,
    *,
    nuisance_mean: torch.Tensor,
    nuisance_factor: torch.Tensor,
    nuisance_measurement_factor: torch.Tensor,
) -> DiffusionBeliefBranch:
    """Append an independent low-rank operator nuisance to one branch.

    The branch measurement mean must already be evaluated at
    ``nuisance_mean``.  ``nuisance_measurement_factor`` is the local forward
    response to the columns of ``nuisance_factor``.
    """
    if branch.state_mean is None or branch.state_factor is None:
        raise ValueError("the image branch must carry aligned state factors")
    nuisance_mean = nuisance_mean.flatten().to(branch.measurement_mean)
    if nuisance_factor.dim() != 2:
        raise ValueError("nuisance_factor must be two-dimensional")
    nuisance_factor = nuisance_factor.to(branch.measurement_mean)
    if nuisance_factor.shape[0] != nuisance_mean.numel():
        raise ValueError("nuisance mean and factor dimensions do not match")
    nuisance_measurement_factor = nuisance_measurement_factor.to(
        branch.measurement_mean
    )
    if (
        nuisance_measurement_factor.dim() != 2
        or nuisance_measurement_factor.shape[0]
        != branch.measurement_mean.numel()
        or nuisance_measurement_factor.shape[1]
        != nuisance_factor.shape[1]
    ):
        raise ValueError(
            "nuisance measurement factor must have shape "
            "(measurement_dimension, nuisance_rank)"
        )
    if (
        not bool(torch.isfinite(nuisance_mean).all())
        or not bool(torch.isfinite(nuisance_factor).all())
        or not bool(torch.isfinite(nuisance_measurement_factor).all())
    ):
        raise ValueError("nuisance tensors must be finite")

    image_state_dimension = branch.state_mean.numel()
    image_rank = branch.rank
    nuisance_rank = int(nuisance_factor.shape[1])
    joint_state_factor = branch.state_factor.new_zeros(
        (
            image_state_dimension + nuisance_mean.numel(),
            image_rank + nuisance_rank,
        )
    )
    joint_state_factor[:image_state_dimension, :image_rank] = (
        branch.state_factor
    )
    joint_state_factor[
        image_state_dimension:, image_rank:
    ] = nuisance_factor
    return DiffusionBeliefBranch(
        weight=branch.weight,
        measurement_mean=branch.measurement_mean,
        measurement_factor=torch.cat(
            (
                branch.measurement_factor,
                nuisance_measurement_factor,
            ),
            dim=1,
        ),
        state_mean=torch.cat((branch.state_mean, nuisance_mean)),
        state_factor=joint_state_factor,
        label=branch.label,
    )


@torch.no_grad()
def finite_difference_nuisance_measurement_factor(
    *,
    state_mean: torch.Tensor,
    nuisance_mean: torch.Tensor,
    nuisance_factor: torch.Tensor,
    forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    fd_step: float = 1e-2,
) -> torch.Tensor:
    """Push physical nuisance factors through a differentiable forward model.

    Each nuisance-factor column is interpreted as a one-standard-deviation
    physical perturbation.  The returned columns are central finite
    differences with respect to the corresponding standardized coefficient.
    This calls only the forward operator, not the diffusion denoiser.
    """
    if fd_step <= 0:
        raise ValueError("fd_step must be positive")
    nuisance_mean = nuisance_mean.flatten()
    if nuisance_factor.dim() != 2:
        raise ValueError("nuisance_factor must be two-dimensional")
    if nuisance_factor.shape[0] != nuisance_mean.numel():
        raise ValueError("nuisance mean and factor dimensions do not match")
    baseline = forward(state_mean, nuisance_mean)
    measurement_dimension = real_flatten(baseline).numel()
    real_dtype = (
        baseline.real.dtype if torch.is_complex(baseline) else baseline.dtype
    )
    output = torch.zeros(
        measurement_dimension,
        nuisance_factor.shape[1],
        dtype=real_dtype,
        device=baseline.device,
    )
    for column in range(nuisance_factor.shape[1]):
        direction = nuisance_factor[:, column]
        plus = forward(state_mean, nuisance_mean + fd_step * direction)
        minus = forward(state_mean, nuisance_mean - fd_step * direction)
        if plus.shape != baseline.shape or minus.shape != baseline.shape:
            raise ValueError("forward output shape changed under nuisance")
        output[:, column] = (
            real_flatten(plus) - real_flatten(minus)
        ) / (2.0 * fd_step)
    return output


def gaussian_target_information(
    branch: DiffusionBeliefBranch,
    indices: Sequence[int] | torch.Tensor,
    noise_variance: float | torch.Tensor,
    *,
    target_state_rows: Sequence[int] | torch.Tensor,
    nuisance_state_rows: Sequence[int] | torch.Tensor,
    rank_tolerance: float = 1e-7,
) -> TargetInformationDiagnostic:
    """Information about target rows under one joint local Gaussian.

    Let ``xi ~ N(0, I)`` be the branch coefficient, ``X = U_x xi`` the target,
    ``Phi = U_phi xi`` the nuisance, and ``Y = V xi + epsilon`` the action.
    The target information follows from the difference between marginal
    measurement entropy and measurement entropy conditional on ``X``.
    """
    if branch.state_mean is None or branch.state_factor is None:
        raise ValueError("branch must carry joint state factors")
    measurement_indices = _indices(
        indices,
        dimension=branch.measurement_mean.numel(),
        device=branch.measurement_mean.device,
    )
    target_rows = _indices(
        target_state_rows,
        dimension=branch.state_mean.numel(),
        device=branch.state_mean.device,
    )
    nuisance_rows = _indices(
        nuisance_state_rows,
        dimension=branch.state_mean.numel(),
        device=branch.state_mean.device,
    )
    noise = _noise(
        noise_variance,
        measurement_indices,
        full_dimension=branch.measurement_mean.numel(),
        reference=branch.measurement_mean,
    )
    action_factor = (
        branch.measurement_factor[measurement_indices]
        / noise.sqrt().unsqueeze(1)
    )
    target_factor = branch.state_factor[target_rows]
    nuisance_factor = branch.state_factor[nuisance_rows]

    joint_logdet = _factor_logdet(action_factor)
    target_null, target_rank = _null_basis(
        target_factor, tolerance=rank_tolerance
    )
    conditional_target_logdet = _factor_logdet(
        action_factor @ target_null
    )
    target_information = 0.5 * (
        joint_logdet - conditional_target_logdet
    )

    nuisance_null, nuisance_rank = _null_basis(
        nuisance_factor, tolerance=rank_tolerance
    )
    conditioned_action = action_factor @ nuisance_null
    conditioned_target = target_factor @ nuisance_null
    residual_target_null, _ = _null_basis(
        conditioned_target, tolerance=rank_tolerance
    )
    known_nuisance_information = 0.5 * (
        _factor_logdet(conditioned_action)
        - _factor_logdet(conditioned_action @ residual_target_null)
    )
    optimism = known_nuisance_information - target_information
    return TargetInformationDiagnostic(
        joint_information=float((0.5 * joint_logdet).item()),
        target_information=float(target_information.item()),
        known_nuisance_information=float(
            known_nuisance_information.item()
        ),
        known_operator_optimism=float(optimism.item()),
        target_rank=target_rank,
        nuisance_rank=nuisance_rank,
    )


def _indices(
    values: Sequence[int] | torch.Tensor,
    *,
    dimension: int,
    device: torch.device,
) -> torch.Tensor:
    result = torch.as_tensor(values, device=device)
    if result.dtype == torch.bool:
        if result.numel() != dimension:
            raise ValueError("boolean index has the wrong length")
        result = torch.nonzero(result, as_tuple=False).flatten()
    else:
        result = result.to(torch.long).flatten()
    if bool(((result < 0) | (result >= dimension)).any()):
        raise ValueError("index outside valid range")
    if torch.unique(result).numel() != result.numel():
        raise ValueError("indices must be unique")
    return result


def _noise(
    value: float | torch.Tensor,
    indices: torch.Tensor,
    *,
    full_dimension: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    noise = torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device
    )
    if noise.numel() == 1:
        noise = noise.expand(indices.numel())
    elif noise.dim() == 1 and noise.numel() == full_dimension:
        noise = noise[indices]
    elif noise.dim() == 1 and noise.numel() == indices.numel():
        pass
    else:
        raise ValueError("noise has incompatible dimensions")
    if bool((noise <= 0).any()) or not bool(torch.isfinite(noise).all()):
        raise ValueError("noise variances must be finite and positive")
    return noise


def _factor_logdet(factor: torch.Tensor) -> torch.Tensor:
    rank = int(factor.shape[1])
    if rank == 0:
        return factor.new_tensor(0.0)
    matrix = (
        torch.eye(rank, dtype=factor.dtype, device=factor.device)
        + factor.T @ factor
    )
    sign, value = torch.linalg.slogdet(matrix)
    if float(sign.item()) <= 0:
        raise RuntimeError("positive factor Gram has nonpositive determinant")
    return value


def _null_basis(
    factor: torch.Tensor,
    *,
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    coefficient_dimension = int(factor.shape[1])
    if coefficient_dimension == 0:
        return factor.new_zeros((0, 0)), 0
    _, singular_values, right = torch.linalg.svd(
        factor, full_matrices=True
    )
    if singular_values.numel():
        threshold = max(
            tolerance,
            tolerance * float(singular_values.max().item()),
        )
        rank = int((singular_values > threshold).sum().item())
    else:
        rank = 0
    return right[rank:].T, rank
