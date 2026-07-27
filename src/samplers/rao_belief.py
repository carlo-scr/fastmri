"""Operator-sufficient Rao--Blackwellized beliefs for active diffusion.

The module represents a high-dimensional inverse-problem belief as a finite
mixture of low-rank local Gaussians,

    b(z) = sum_k w_k N(m_k, U_k U_k^T).

Each branch may also carry the aligned factor of a complete candidate
measurement bank.  This is the object used by RaoSense:

* branch means retain between-trajectory (mode) uncertainty;
* low-rank factors retain within-trajectory Tweedie/JVP uncertainty;
* the law of total covariance combines both without drawing endpoint samples;
* every action is whitened and compressed to its signal-bearing subspace;
* likelihoods, mutual information, fantasies, and Gaussian updates are then
  evaluated in at most ``sum_k rank(U_k) + K - 1`` coordinates.

The Gaussian-mixture update and non-myopic lookahead are classical tools.  The
intended research contribution is the diffusion-derived, operator-sufficient
belief that makes those tools practical for very large candidate operators.

All tensors in this module use real coordinates.  Complex measurements must
first be converted with ``operator_sketch.real_flatten`` so that proper
complex noise is represented by one variance for each real component.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

import torch

from .operator_sketch import OperatorCovarianceSketch, real_flatten


def _as_real_vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if torch.is_complex(value):
        raise ValueError(f"{name} must use stacked real coordinates")
    if value.dim() != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _as_real_factor(
    value: torch.Tensor,
    *,
    rows: int,
    name: str,
) -> torch.Tensor:
    if torch.is_complex(value):
        raise ValueError(f"{name} must use stacked real coordinates")
    if value.dim() != 2 or value.shape[0] != rows:
        raise ValueError(f"{name} must have shape ({rows}, rank)")
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return value


def _normalize_indices(
    indices: Sequence[int] | torch.Tensor,
    *,
    dimension: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.as_tensor(indices, device=device)
    if values.dtype == torch.bool:
        if values.dim() != 1 or values.numel() != dimension:
            raise ValueError("boolean action mask has the wrong length")
        values = torch.nonzero(values, as_tuple=False).flatten()
    else:
        values = values.to(torch.long).flatten()
    if values.numel() == 0:
        return values
    if bool(((values < 0) | (values >= dimension)).any()):
        raise ValueError("action index is outside the measurement bank")
    if torch.unique(values).numel() != values.numel():
        raise ValueError("action indices must be unique")
    return values


def _selected_noise(
    noise_variance: float | torch.Tensor,
    indices: torch.Tensor,
    *,
    full_dimension: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    noise = torch.as_tensor(noise_variance, dtype=dtype, device=device)
    if noise.numel() == 1:
        selected = noise.expand(indices.numel())
    elif noise.dim() == 1 and noise.numel() == full_dimension:
        selected = noise[indices]
    elif noise.dim() == 1 and noise.numel() == indices.numel():
        selected = noise
    else:
        raise ValueError(
            "noise_variance must be scalar, measurement-sized, or action-sized"
        )
    if not bool(torch.isfinite(selected).all()) or bool((selected <= 0).any()):
        raise ValueError("noise variances must be finite and strictly positive")
    return selected


def _logdet_identity_plus_factor(factor: torch.Tensor) -> torch.Tensor:
    """Return ``log det(I + factor factor^T)`` through the small Gram."""
    rank = int(factor.shape[1])
    if rank == 0:
        return factor.new_tensor(0.0)
    gram = torch.eye(rank, dtype=factor.dtype, device=factor.device)
    gram = gram + factor.T @ factor
    sign, value = torch.linalg.slogdet(gram)
    if float(sign.item()) <= 0:
        raise RuntimeError("positive-definite factor Gram has nonpositive sign")
    return value


def _covariance_shrink(factor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior coefficient covariance and a protected square-root factor."""
    rank = int(factor.shape[1])
    if rank == 0:
        empty = factor.new_zeros((0, 0))
        return empty, empty
    precision = (
        torch.eye(rank, dtype=factor.dtype, device=factor.device)
        + factor.T @ factor
    )
    covariance = torch.linalg.inv(precision)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = 32.0 * torch.finfo(covariance.dtype).eps
    eigenvalues = eigenvalues.clamp_min(floor)
    square_root = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.T
    return covariance, square_root


def _gaussian_factor_logpdf(
    samples: torch.Tensor,
    mean: torch.Tensor,
    factor: torch.Tensor,
) -> torch.Tensor:
    """Log density of ``N(mean, I + factor factor^T)``.

    The Woodbury identity keeps the expensive solve in the branch rank rather
    than the action dimension.
    """
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    if samples.dim() != 2 or samples.shape[1] != mean.numel():
        raise ValueError("samples and projected mean have incompatible shapes")
    difference = samples - mean.unsqueeze(0)
    dimension = int(mean.numel())
    rank = int(factor.shape[1])
    quadratic = difference.square().sum(dim=1)
    logdet = difference.new_tensor(0.0)
    if rank:
        small = (
            torch.eye(rank, dtype=factor.dtype, device=factor.device)
            + factor.T @ factor
        )
        chol = torch.linalg.cholesky(small)
        projected = difference @ factor
        solved = torch.cholesky_solve(projected.T, chol).T
        quadratic = quadratic - (projected * solved).sum(dim=1)
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    constant = dimension * math.log(2.0 * math.pi)
    return -0.5 * (constant + logdet + quadratic)


@dataclass(frozen=True)
class DiffusionBeliefBranch:
    """One local Gaussian branch in state and candidate-measurement space.

    ``state_factor`` and ``measurement_factor`` have the same column
    semantics.  This alignment permits an observation to update both spaces
    through one small coefficient posterior.
    """

    weight: float
    measurement_mean: torch.Tensor
    measurement_factor: torch.Tensor
    state_mean: torch.Tensor | None = None
    state_factor: torch.Tensor | None = None
    label: str = ""

    def __post_init__(self) -> None:
        measurement_mean = _as_real_vector(
            self.measurement_mean, name="measurement_mean"
        )
        measurement_factor = _as_real_factor(
            self.measurement_factor,
            rows=measurement_mean.numel(),
            name="measurement_factor",
        )
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0:
            raise ValueError("branch weight must be finite and nonnegative")
        if measurement_factor.device != measurement_mean.device:
            raise ValueError("measurement mean and factor must share a device")
        if measurement_factor.dtype != measurement_mean.dtype:
            raise ValueError("measurement mean and factor must share a dtype")
        if (self.state_mean is None) != (self.state_factor is None):
            raise ValueError("state_mean and state_factor must be supplied together")
        if self.state_mean is not None and self.state_factor is not None:
            state_mean = _as_real_vector(self.state_mean, name="state_mean")
            state_factor = _as_real_factor(
                self.state_factor,
                rows=state_mean.numel(),
                name="state_factor",
            )
            if state_factor.shape[1] != measurement_factor.shape[1]:
                raise ValueError(
                    "state and measurement factors must have the same rank"
                )
            if (
                state_mean.device != measurement_mean.device
                or state_mean.dtype != measurement_mean.dtype
                or state_factor.device != measurement_mean.device
                or state_factor.dtype != measurement_mean.dtype
            ):
                raise ValueError("all branch tensors must share dtype and device")

    @property
    def rank(self) -> int:
        return int(self.measurement_factor.shape[1])


@dataclass(frozen=True)
class InformationBounds:
    """Analytic mutual-information bounds for one action."""

    lower: float
    upper: float
    gaussian_upper: float
    mode_upper: float
    entropy_cap: float
    sufficient_rank: int
    scalar_measurements: int


@dataclass(frozen=True)
class InformationEstimate:
    """Monte Carlo estimate of exact mixture information under the belief."""

    total: float
    within: float
    mode: float
    standard_error: float
    num_samples: int
    sufficient_rank: int


@dataclass(frozen=True)
class ProjectedAction:
    """Noise-whitened sufficient representation of one candidate action."""

    indices: torch.Tensor
    basis: torch.Tensor
    reference_mean: torch.Tensor
    noise_standard_deviation: torch.Tensor
    weights: torch.Tensor
    component_means: tuple[torch.Tensor, ...]
    component_factors: tuple[torch.Tensor, ...]
    numerical_rank_bound: int

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[1])

    def project_observation(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.to(
            dtype=self.reference_mean.dtype,
            device=self.reference_mean.device,
        ).flatten()
        if observation.numel() != self.indices.numel():
            raise ValueError("observation must contain one value per action index")
        whitened = (
            observation - self.reference_mean
        ) / self.noise_standard_deviation
        return self.basis.T @ whitened

    def sample(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample sufficient outcomes and their component labels."""
        if count <= 0:
            raise ValueError("sample count must be positive")
        labels = torch.multinomial(
            self.weights.detach().cpu(),
            count,
            replacement=True,
            generator=generator,
        ).to(self.weights.device)
        output = self.basis.new_empty((count, self.dimension))
        for row in range(count):
            component = int(labels[row].item())
            mean = self.component_means[component]
            factor = self.component_factors[component]
            noise = torch.randn(
                self.dimension,
                generator=generator,
                device="cpu",
                dtype=self.basis.dtype,
            ).to(self.basis.device)
            if factor.shape[1]:
                coefficient = torch.randn(
                    factor.shape[1],
                    generator=generator,
                    device="cpu",
                    dtype=self.basis.dtype,
                ).to(self.basis.device)
                noise = noise + factor @ coefficient
            output[row] = mean + noise
        return output, labels

    def component_logpdf(
        self,
        samples: torch.Tensor,
        component: int,
    ) -> torch.Tensor:
        return _gaussian_factor_logpdf(
            samples,
            self.component_means[component],
            self.component_factors[component],
        )

    def mixture_logpdf(self, samples: torch.Tensor) -> torch.Tensor:
        terms = []
        tiny = torch.finfo(self.weights.dtype).tiny
        for component in range(len(self.component_means)):
            terms.append(
                self.component_logpdf(samples, component)
                + self.weights[component].clamp_min(tiny).log()
            )
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


@dataclass(frozen=True)
class ActionEvaluation:
    action: int
    bounds: InformationBounds
    estimate: InformationEstimate | None
    safely_pruned: bool


@dataclass(frozen=True)
class LookaheadPlan:
    action: int
    objective: float
    immediate_information: float
    expected_future_information: float
    evaluations: tuple[ActionEvaluation, ...]
    beam: tuple[int, ...]
    diagnostics: Mapping[str, float | int]


class RaoBlackwellizedBelief:
    """Validated finite mixture with aligned low-rank branch factors."""

    def __init__(self, branches: Sequence[DiffusionBeliefBranch]):
        if not branches:
            raise ValueError("at least one branch is required")
        first = branches[0]
        measurement_dimension = first.measurement_mean.numel()
        dtype = first.measurement_mean.dtype
        device = first.measurement_mean.device
        has_state = first.state_mean is not None
        state_dimension = (
            int(first.state_mean.numel()) if first.state_mean is not None else None
        )
        for branch in branches:
            if branch.measurement_mean.numel() != measurement_dimension:
                raise ValueError("all branches must share a measurement dimension")
            if (
                branch.measurement_mean.dtype != dtype
                or branch.measurement_mean.device != device
            ):
                raise ValueError("all branches must share dtype and device")
            if (branch.state_mean is not None) != has_state:
                raise ValueError("either every branch carries state factors or none")
            if (
                has_state
                and branch.state_mean is not None
                and branch.state_mean.numel() != state_dimension
            ):
                raise ValueError("all branches must share a state dimension")
        weights = torch.tensor(
            [float(branch.weight) for branch in branches],
            dtype=dtype,
            device=device,
        )
        total = weights.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0:
            raise ValueError("branch weights must have positive finite total")
        weights = weights / total
        self._branches = tuple(
            replace(branch, weight=float(weights[index].item()))
            for index, branch in enumerate(branches)
        )
        self._weights = weights
        self._measurement_dimension = int(measurement_dimension)
        self._state_dimension = state_dimension

    @property
    def branches(self) -> tuple[DiffusionBeliefBranch, ...]:
        return self._branches

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @property
    def num_branches(self) -> int:
        return len(self._branches)

    @property
    def measurement_dimension(self) -> int:
        return self._measurement_dimension

    @property
    def state_dimension(self) -> int | None:
        return self._state_dimension

    @property
    def maximum_sufficient_rank(self) -> int:
        return max(
            0,
            sum(branch.rank for branch in self.branches)
            + self.num_branches
            - 1,
        )

    def categorical_entropy(self) -> torch.Tensor:
        positive = self.weights > 0
        return -(self.weights[positive] * self.weights[positive].log()).sum()

    def measurement_moments(
        self,
        indices: Sequence[int] | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact mixture mean and a total-covariance factor."""
        if indices is None:
            selected = torch.arange(
                self.measurement_dimension,
                device=self.weights.device,
            )
        else:
            selected = _normalize_indices(
                indices,
                dimension=self.measurement_dimension,
                device=self.weights.device,
            )
        means = torch.stack(
            [branch.measurement_mean[selected] for branch in self.branches],
            dim=0,
        )
        mean = (self.weights.unsqueeze(1) * means).sum(dim=0)
        columns = []
        for index, branch in enumerate(self.branches):
            scale = self.weights[index].sqrt()
            if branch.rank:
                columns.append(scale * branch.measurement_factor[selected])
            columns.append(scale * (means[index] - mean).unsqueeze(1))
        factor = torch.cat(columns, dim=1) if columns else mean.new_zeros(
            (mean.numel(), 0)
        )
        return mean, factor

    def state_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.state_dimension is None:
            raise ValueError("this belief does not carry state factors")
        means = torch.stack(
            [
                branch.state_mean
                for branch in self.branches
                if branch.state_mean is not None
            ],
            dim=0,
        )
        mean = (self.weights.unsqueeze(1) * means).sum(dim=0)
        columns = []
        for index, branch in enumerate(self.branches):
            assert branch.state_mean is not None
            assert branch.state_factor is not None
            scale = self.weights[index].sqrt()
            if branch.rank:
                columns.append(scale * branch.state_factor)
            columns.append(scale * (branch.state_mean - mean).unsqueeze(1))
        return mean, torch.cat(columns, dim=1)

    def information_bounds(
        self,
        indices: Sequence[int] | torch.Tensor,
        noise_variance: float | torch.Tensor,
    ) -> InformationBounds:
        selected = _normalize_indices(
            indices,
            dimension=self.measurement_dimension,
            device=self.weights.device,
        )
        dtype = self.branches[0].measurement_mean.dtype
        noise = _selected_noise(
            noise_variance,
            selected,
            full_dimension=self.measurement_dimension,
            dtype=dtype,
            device=self.weights.device,
        )
        inverse_standard_deviation = noise.rsqrt()
        within = self.weights.new_tensor(0.0)
        for index, branch in enumerate(self.branches):
            factor = (
                inverse_standard_deviation.unsqueeze(1)
                * branch.measurement_factor[selected]
            )
            within = within + 0.5 * self.weights[index] * (
                _logdet_identity_plus_factor(factor)
            )
        _, total_factor = self.measurement_moments(selected)
        whitened_total = inverse_standard_deviation.unsqueeze(1) * total_factor
        gaussian_upper = 0.5 * _logdet_identity_plus_factor(whitened_total)
        # Roundoff can make the moment upper bound microscopically smaller.
        gaussian_upper = torch.maximum(gaussian_upper, within)
        entropy = self.categorical_entropy()
        mode_upper = torch.minimum(gaussian_upper - within, entropy)
        upper = within + mode_upper
        sufficient_rank = _numerical_rank(whitened_total)
        return InformationBounds(
            lower=float(within.item()),
            upper=float(upper.item()),
            gaussian_upper=float(gaussian_upper.item()),
            mode_upper=float(mode_upper.item()),
            entropy_cap=float(entropy.item()),
            sufficient_rank=sufficient_rank,
            scalar_measurements=int(selected.numel()),
        )

    def project_action(
        self,
        indices: Sequence[int] | torch.Tensor,
        noise_variance: float | torch.Tensor,
        *,
        relative_tolerance: float = 1e-6,
        absolute_tolerance: float = 1e-10,
    ) -> ProjectedAction:
        """Construct the noise-whitened signal-bearing action subspace."""
        selected = _normalize_indices(
            indices,
            dimension=self.measurement_dimension,
            device=self.weights.device,
        )
        dtype = self.branches[0].measurement_mean.dtype
        noise = _selected_noise(
            noise_variance,
            selected,
            full_dimension=self.measurement_dimension,
            dtype=dtype,
            device=self.weights.device,
        )
        standard_deviation = noise.sqrt()
        reference_mean, total_factor = self.measurement_moments(selected)
        whitened_total = total_factor / standard_deviation.unsqueeze(1)
        if whitened_total.numel() == 0:
            basis = whitened_total.new_zeros((selected.numel(), 0))
        else:
            left, singular_values, _ = torch.linalg.svd(
                whitened_total, full_matrices=False
            )
            if singular_values.numel() == 0:
                basis = left[:, :0]
            else:
                threshold = max(
                    float(absolute_tolerance),
                    float(relative_tolerance)
                    * float(singular_values.max().item()),
                )
                basis = left[:, singular_values > threshold]
        component_means = []
        component_factors = []
        for branch in self.branches:
            whitened_mean = (
                branch.measurement_mean[selected] - reference_mean
            ) / standard_deviation
            whitened_factor = (
                branch.measurement_factor[selected]
                / standard_deviation.unsqueeze(1)
            )
            component_means.append(basis.T @ whitened_mean)
            component_factors.append(basis.T @ whitened_factor)
        return ProjectedAction(
            indices=selected,
            basis=basis,
            reference_mean=reference_mean,
            noise_standard_deviation=standard_deviation,
            weights=self.weights,
            component_means=tuple(component_means),
            component_factors=tuple(component_factors),
            numerical_rank_bound=self.maximum_sufficient_rank,
        )

    def estimate_information(
        self,
        indices: Sequence[int] | torch.Tensor,
        noise_variance: float | torch.Tensor,
        *,
        num_samples: int = 256,
        generator: torch.Generator | None = None,
    ) -> InformationEstimate:
        """Estimate exact-under-belief MI after sufficient projection.

        The within-component term is analytic.  Monte Carlo is used only for
        the bounded categorical mode-information term.
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        bounds = self.information_bounds(indices, noise_variance)
        projection = self.project_action(indices, noise_variance)
        if projection.dimension == 0 or self.num_branches == 1:
            return InformationEstimate(
                total=bounds.lower,
                within=bounds.lower,
                mode=0.0,
                standard_error=0.0,
                num_samples=num_samples,
                sufficient_rank=projection.dimension,
            )
        samples, labels = projection.sample(num_samples, generator=generator)
        mixture_logpdf = projection.mixture_logpdf(samples)
        selected_logpdf = samples.new_empty((num_samples,))
        for component in range(self.num_branches):
            rows = labels == component
            if bool(rows.any()):
                selected_logpdf[rows] = projection.component_logpdf(
                    samples[rows], component
                )
        mode_samples = selected_logpdf - mixture_logpdf
        mode_information = mode_samples.mean()
        standard_error = (
            mode_samples.std(unbiased=True)
            / math.sqrt(num_samples)
            if num_samples > 1
            else mode_samples.new_tensor(float("nan"))
        )
        return InformationEstimate(
            total=float(bounds.lower + mode_information.item()),
            within=bounds.lower,
            mode=float(mode_information.item()),
            standard_error=float(standard_error.item()),
            num_samples=num_samples,
            sufficient_rank=projection.dimension,
        )

    def condition_projected(
        self,
        projection: ProjectedAction,
        outcome: torch.Tensor,
    ) -> "RaoBlackwellizedBelief":
        """Condition every branch on one sufficient-coordinate outcome."""
        outcome = outcome.to(
            dtype=self.weights.dtype, device=self.weights.device
        ).flatten()
        if outcome.numel() != projection.dimension:
            raise ValueError("projected outcome has the wrong dimension")
        if len(projection.component_means) != self.num_branches:
            raise ValueError("projection was built for a different belief")
        log_weights = []
        updated = []
        tiny = torch.finfo(self.weights.dtype).tiny
        for index, branch in enumerate(self.branches):
            factor = projection.component_factors[index]
            innovation = outcome - projection.component_means[index]
            coefficient_covariance, square_root = _covariance_shrink(factor)
            if branch.rank:
                coefficient_mean = (
                    coefficient_covariance @ factor.T @ innovation
                )
                measurement_mean = (
                    branch.measurement_mean
                    + branch.measurement_factor @ coefficient_mean
                )
                measurement_factor = branch.measurement_factor @ square_root
                if branch.state_mean is not None and branch.state_factor is not None:
                    state_mean = (
                        branch.state_mean
                        + branch.state_factor @ coefficient_mean
                    )
                    state_factor = branch.state_factor @ square_root
                else:
                    state_mean = None
                    state_factor = None
            else:
                measurement_mean = branch.measurement_mean
                measurement_factor = branch.measurement_factor
                state_mean = branch.state_mean
                state_factor = branch.state_factor
            log_likelihood = _gaussian_factor_logpdf(
                outcome,
                projection.component_means[index],
                factor,
            )[0]
            log_weights.append(
                self.weights[index].clamp_min(tiny).log() + log_likelihood
            )
            updated.append(
                DiffusionBeliefBranch(
                    weight=1.0,
                    measurement_mean=measurement_mean,
                    measurement_factor=measurement_factor,
                    state_mean=state_mean,
                    state_factor=state_factor,
                    label=branch.label,
                )
            )
        normalized_weights = torch.softmax(torch.stack(log_weights), dim=0)
        return RaoBlackwellizedBelief(
            [
                replace(branch, weight=float(normalized_weights[index].item()))
                for index, branch in enumerate(updated)
            ]
        )

    def condition(
        self,
        indices: Sequence[int] | torch.Tensor,
        observation: torch.Tensor,
        noise_variance: float | torch.Tensor,
    ) -> "RaoBlackwellizedBelief":
        projection = self.project_action(indices, noise_variance)
        return self.condition_projected(
            projection,
            projection.project_observation(observation),
        )


def branch_from_operator_sketch(
    *,
    weight: float,
    measurement_mean: torch.Tensor,
    state_mean: torch.Tensor,
    sketch: OperatorCovarianceSketch,
    label: str = "",
) -> DiffusionBeliefBranch:
    """Create an aligned belief branch from one protected operator sketch."""
    measurement_vector = real_flatten(measurement_mean)
    state_vector = real_flatten(state_mean)
    if measurement_vector.numel() != sketch.measurement_factor.shape[0]:
        raise ValueError("measurement mean does not match the operator sketch")
    if state_vector.numel() != sketch.image_factor.shape[0]:
        raise ValueError("state mean does not match the operator sketch")
    return DiffusionBeliefBranch(
        weight=weight,
        measurement_mean=measurement_vector,
        measurement_factor=sketch.measurement_factor,
        state_mean=state_vector,
        state_factor=sketch.image_factor,
        label=label,
    )


def _numerical_rank(matrix: torch.Tensor) -> int:
    if matrix.numel() == 0:
        return 0
    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() == 0:
        return 0
    threshold = max(
        1e-10,
        1e-6 * float(singular_values.max().item()),
    )
    return int((singular_values > threshold).sum().item())


def screen_actions(
    belief: RaoBlackwellizedBelief,
    actions: Mapping[int, Sequence[int] | torch.Tensor],
    noise_variance: float | torch.Tensor | Mapping[int, float | torch.Tensor],
    *,
    tolerance: float = 1e-10,
) -> tuple[tuple[int, ...], tuple[ActionEvaluation, ...]]:
    """Safely prune actions whose MI upper bound loses to another lower bound."""
    if not actions:
        raise ValueError("at least one action is required")
    bounds = {
        action: belief.information_bounds(
            indices,
            _action_noise(noise_variance, action),
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
        ActionEvaluation(
            action=action,
            bounds=bounds[action],
            estimate=None,
            safely_pruned=action not in survivors,
        )
        for action in sorted(actions)
    )
    return survivors, evaluations


def plan_horizon_two(
    belief: RaoBlackwellizedBelief,
    actions: Mapping[int, Sequence[int] | torch.Tensor],
    noise_variance: float | torch.Tensor | Mapping[int, float | torch.Tensor],
    *,
    beam_width: int = 8,
    num_fantasies: int = 16,
    information_samples: int = 128,
    future_information_samples: int | None = None,
    discount: float = 1.0,
    action_costs: Mapping[int, float] | None = None,
    successors: Mapping[int, Sequence[int]] | None = None,
    generator: torch.Generator | None = None,
) -> LookaheadPlan:
    """Bound-screened, outcome-dependent, receding-horizon-two planning.

    The routine performs no denoiser evaluation.  It samples fantasy outcomes,
    reconditions the compressed branch factors analytically, and returns only
    the root action.  A real system should acquire that action, condition on
    the actual observation, and replan.
    """
    if beam_width <= 0 or num_fantasies <= 0 or information_samples <= 0:
        raise ValueError("beam_width, fantasies, and samples must be positive")
    if discount < 0:
        raise ValueError("discount must be nonnegative")
    if future_information_samples is None:
        future_information_samples = information_samples
    if future_information_samples <= 0:
        raise ValueError("future_information_samples must be positive")
    costs = action_costs or {}

    survivors, initial_evaluations = screen_actions(
        belief, actions, noise_variance
    )
    estimates: dict[int, InformationEstimate] = {}
    evaluation_map = {row.action: row for row in initial_evaluations}
    for action in survivors:
        estimates[action] = belief.estimate_information(
            actions[action],
            _action_noise(noise_variance, action),
            num_samples=information_samples,
            generator=generator,
        )
        evaluation_map[action] = replace(
            evaluation_map[action], estimate=estimates[action]
        )
    beam = tuple(
        sorted(
            survivors,
            key=lambda action: (
                estimates[action].total - float(costs.get(action, 0.0)),
                -action,
            ),
            reverse=True,
        )[:beam_width]
    )
    if not beam:
        raise RuntimeError("information bounds pruned every action")

    root_values: dict[int, tuple[float, float, float]] = {}
    expanded_nodes = 0
    pruned_nodes = sum(row.safely_pruned for row in initial_evaluations)
    for action in beam:
        action_noise = _action_noise(noise_variance, action)
        projection = belief.project_action(actions[action], action_noise)
        fantasies, _ = projection.sample(num_fantasies, generator=generator)
        future_values = []
        if successors is None:
            allowed = tuple(candidate for candidate in actions if candidate != action)
        else:
            allowed = tuple(
                candidate
                for candidate in successors.get(action, ())
                if candidate in actions and candidate != action
            )
        for fantasy in fantasies:
            if not allowed:
                future_values.append(0.0)
                continue
            posterior = belief.condition_projected(projection, fantasy)
            future_actions = {candidate: actions[candidate] for candidate in allowed}
            future_survivors, future_rows = screen_actions(
                posterior, future_actions, noise_variance
            )
            pruned_nodes += sum(row.safely_pruned for row in future_rows)
            best_future = float("-inf")
            for candidate in future_survivors:
                expanded_nodes += 1
                estimate = posterior.estimate_information(
                    actions[candidate],
                    _action_noise(noise_variance, candidate),
                    num_samples=future_information_samples,
                    generator=generator,
                )
                value = estimate.total - float(costs.get(candidate, 0.0))
                best_future = max(best_future, value)
            future_values.append(max(0.0, best_future))
        future = float(sum(future_values) / len(future_values))
        immediate = (
            estimates[action].total - float(costs.get(action, 0.0))
        )
        root_values[action] = (
            immediate + discount * future,
            immediate,
            future,
        )
    selected = max(
        beam,
        key=lambda action: (root_values[action][0], -action),
    )
    objective, immediate, future = root_values[selected]
    return LookaheadPlan(
        action=selected,
        objective=objective,
        immediate_information=immediate,
        expected_future_information=future,
        evaluations=tuple(evaluation_map[action] for action in sorted(actions)),
        beam=beam,
        diagnostics={
            "num_actions": len(actions),
            "root_survivors": len(survivors),
            "beam_width": len(beam),
            "num_fantasies": int(num_fantasies),
            "expanded_future_nodes": int(expanded_nodes),
            "safely_pruned_nodes": int(pruned_nodes),
            "maximum_sufficient_rank": belief.maximum_sufficient_rank,
        },
    )


def _action_noise(
    noise_variance: float | torch.Tensor | Mapping[int, float | torch.Tensor],
    action: int,
) -> float | torch.Tensor:
    if isinstance(noise_variance, Mapping):
        if action not in noise_variance:
            raise KeyError(f"missing noise variance for action {action}")
        return noise_variance[action]
    return noise_variance
