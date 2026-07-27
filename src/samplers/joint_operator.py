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

For a finite branch mixture, the H1 target utility treats the target as
``(branch label, selected target-state rows)``.  It marginalizes nuisance
directions analytically within each branch, estimates only branch-label
information in the ordinary RaoSense sufficient subspace, and exposes
certified lower/upper bounds for safe one-step screening.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch

from .operator_sketch import real_flatten
from .rao_belief import DiffusionBeliefBranch, RaoBlackwellizedBelief

_ActionNoise = (
    float | torch.Tensor | Mapping[int, float | torch.Tensor]
)


@dataclass(frozen=True)
class TargetInformationDiagnostic:
    joint_information: float
    target_information: float
    known_nuisance_information: float
    known_operator_optimism: float
    target_rank: int
    nuisance_rank: int


@dataclass(frozen=True)
class MixtureTargetInformationBounds:
    """Analytic bounds on ``I((Z, X_target); Y_action)``.

    The lower bound is the weight-averaged, within-branch Gaussian target
    information.  The upper bound adds at most the smaller of categorical
    branch entropy and the remaining joint moment-Gaussian information.
    """

    lower: float
    upper: float
    gaussian_upper: float
    mode_upper: float
    entropy_cap: float
    sufficient_rank: int
    scalar_measurements: int


@dataclass(frozen=True)
class MixtureTargetInformationEstimate:
    """Monte Carlo estimate of ``I((Z, X_target); Y_action)``."""

    total: float
    within_target: float
    mode: float
    standard_error: float
    num_samples: int
    sufficient_rank: int


@dataclass(frozen=True)
class TargetActionEvaluation:
    """One H1 target-information screening decision."""

    action: int
    bounds: MixtureTargetInformationBounds
    safely_pruned: bool


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
    if not math.isfinite(rank_tolerance) or rank_tolerance < 0:
        raise ValueError("rank_tolerance must be finite and nonnegative")
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


def mixture_target_information_bounds(
    belief: RaoBlackwellizedBelief,
    indices: Sequence[int] | torch.Tensor,
    noise_variance: float | torch.Tensor,
    *,
    target_state_rows: Sequence[int] | torch.Tensor,
    rank_tolerance: float = 1e-7,
) -> MixtureTargetInformationBounds:
    """Bound target information while marginalizing within-branch nuisance.

    The target is explicitly the pair ``(Z, X_target)``.  Consequently,
    information that distinguishes branch labels counts as useful even when
    two branches induce overlapping target-state distributions.  Callers
    should not use this objective when ``Z`` encodes nuisance-only hypotheses
    that are not part of the desired target.

    Within a branch, every coefficient direction not identified by
    ``target_state_rows`` is marginalized as nuisance.  Across branches, the
    ordinary RaoSense predictive mixture supplies ``I(Z; Y)``.  The returned
    upper bound is

    ``min(joint_moment_gaussian_upper, lower + H(Z))``.

    The joint moment bound includes nuisance-driven predictive variance and
    can therefore be loose, but it remains a valid upper bound on target
    information under the frozen mixture surrogate.
    """
    if belief.state_dimension is None:
        raise ValueError("belief must carry aligned state factors")
    if not math.isfinite(rank_tolerance) or rank_tolerance < 0:
        raise ValueError("rank_tolerance must be finite and nonnegative")

    within_target = 0.0
    for weight, branch in zip(belief.weights, belief.branches):
        branch_weight = float(weight.item())
        if branch_weight == 0.0:
            continue
        diagnostic = gaussian_target_information(
            branch,
            indices,
            noise_variance,
            target_state_rows=target_state_rows,
            # Target information already marginalizes every coefficient
            # direction in the target factor's null space.  No explicit
            # nuisance rows are needed for this quantity.
            nuisance_state_rows=(),
            rank_tolerance=rank_tolerance,
        )
        within_target += branch_weight * diagnostic.target_information

    joint_bounds = belief.information_bounds(indices, noise_variance)
    # In exact arithmetic the full-state moment upper bound dominates target
    # information.  Protect that ordering against null-space/logdet roundoff.
    gaussian_upper = max(joint_bounds.gaussian_upper, within_target)
    entropy = joint_bounds.entropy_cap
    mode_upper = min(max(0.0, gaussian_upper - within_target), entropy)
    upper = within_target + mode_upper
    return MixtureTargetInformationBounds(
        lower=within_target,
        upper=upper,
        gaussian_upper=gaussian_upper,
        mode_upper=mode_upper,
        entropy_cap=entropy,
        sufficient_rank=joint_bounds.sufficient_rank,
        scalar_measurements=joint_bounds.scalar_measurements,
    )


def estimate_mixture_target_information(
    belief: RaoBlackwellizedBelief,
    indices: Sequence[int] | torch.Tensor,
    noise_variance: float | torch.Tensor,
    *,
    target_state_rows: Sequence[int] | torch.Tensor,
    rank_tolerance: float = 1e-7,
    num_samples: int = 256,
    generator: torch.Generator | None = None,
) -> MixtureTargetInformationEstimate:
    """Estimate ``I((Z, X_target); Y_action)`` under the mixture belief.

    Within-branch target information is analytic.  Monte Carlo is used only
    for the same projected branch-label term ``I(Z; Y)`` as ordinary
    RaoSense.  The estimate is unbiased under the exact frozen surrogate and
    exact sufficient projection; ``standard_error`` covers only that mode
    Monte Carlo term and is not a selection-adjusted confidence interval.
    """
    bounds = mixture_target_information_bounds(
        belief,
        indices,
        noise_variance,
        target_state_rows=target_state_rows,
        rank_tolerance=rank_tolerance,
    )
    joint_estimate = belief.estimate_information(
        indices,
        noise_variance,
        num_samples=num_samples,
        generator=generator,
    )
    return MixtureTargetInformationEstimate(
        total=bounds.lower + joint_estimate.mode,
        within_target=bounds.lower,
        mode=joint_estimate.mode,
        standard_error=joint_estimate.standard_error,
        num_samples=joint_estimate.num_samples,
        sufficient_rank=joint_estimate.sufficient_rank,
    )


def screen_target_information_actions(
    belief: RaoBlackwellizedBelief,
    actions: Mapping[int, Sequence[int] | torch.Tensor],
    noise_variance: _ActionNoise,
    *,
    target_state_rows: Sequence[int] | torch.Tensor,
    rank_tolerance: float = 1e-7,
    tolerance: float = 1e-10,
) -> tuple[tuple[int, ...], tuple[TargetActionEvaluation, ...]]:
    """Safely prune H1 actions using target-information bounds.

    An action is removed only when its target-information upper bound is
    strictly below another action's lower bound (up to ``tolerance``).  This
    certificate is for the frozen-surrogate, one-step, zero-cost information
    objective only; it is not a horizon-two or reconstruction-utility bound.
    """
    if not actions:
        raise ValueError("at least one action is required")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    bounds = {
        action: mixture_target_information_bounds(
            belief,
            indices,
            _action_noise(noise_variance, action),
            target_state_rows=target_state_rows,
            rank_tolerance=rank_tolerance,
        )
        for action, indices in sorted(actions.items())
    }
    best_lower = max(value.lower for value in bounds.values())
    survivors = tuple(
        action
        for action in sorted(actions)
        if bounds[action].upper + tolerance >= best_lower
    )
    evaluations = tuple(
        TargetActionEvaluation(
            action=action,
            bounds=bounds[action],
            safely_pruned=action not in survivors,
        )
        for action in sorted(actions)
    )
    return survivors, evaluations


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


def _action_noise(
    noise_variance: _ActionNoise,
    action: int,
) -> float | torch.Tensor:
    if isinstance(noise_variance, Mapping):
        if action not in noise_variance:
            raise KeyError(f"missing noise variance for action {action}")
        return noise_variance[action]
    return noise_variance
