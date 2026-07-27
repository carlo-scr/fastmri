import math

import torch

from src.samplers.rao_belief import (
    DiffusionBeliefBranch,
    RaoBlackwellizedBelief,
    plan_horizon_two,
    screen_actions,
)
from src.samplers.mri_forward import fft2c
from src.samplers.multicoil_pv import run_raosense_active_mc
from src.samplers.sense import MultiCoilSENSE


def _branch(weight, mean, factor, state_mean=None, state_factor=None, label=""):
    return DiffusionBeliefBranch(
        weight=weight,
        measurement_mean=torch.as_tensor(mean, dtype=torch.float64),
        measurement_factor=torch.as_tensor(factor, dtype=torch.float64),
        state_mean=(
            torch.as_tensor(state_mean, dtype=torch.float64)
            if state_mean is not None
            else None
        ),
        state_factor=(
            torch.as_tensor(state_factor, dtype=torch.float64)
            if state_factor is not None
            else None
        ),
        label=label,
    )


def test_total_covariance_factor_matches_finite_mixture_exactly():
    branches = [
        _branch(
            0.3,
            [1.0, -0.5, 0.2],
            [[0.7, 0.0], [0.1, 0.4], [0.2, -0.3]],
            state_mean=[0.5, -1.0],
            state_factor=[[0.4, 0.1], [0.2, 0.6]],
        ),
        _branch(
            0.7,
            [-0.2, 0.8, 1.1],
            [[0.2], [0.6], [-0.1]],
            state_mean=[-0.4, 0.3],
            state_factor=[[0.8], [-0.2]],
        ),
    ]
    belief = RaoBlackwellizedBelief(branches)

    mean, factor = belief.measurement_moments()
    exact = torch.zeros(3, 3, dtype=torch.float64)
    for weight, branch in zip(belief.weights, belief.branches):
        offset = branch.measurement_mean - mean
        exact = exact + weight * (
            branch.measurement_factor @ branch.measurement_factor.T
            + torch.outer(offset, offset)
        )
    torch.testing.assert_close(factor @ factor.T, exact)

    state_mean, state_factor = belief.state_moments()
    exact_state = torch.zeros(2, 2, dtype=torch.float64)
    for weight, branch in zip(belief.weights, belief.branches):
        assert branch.state_mean is not None
        assert branch.state_factor is not None
        offset = branch.state_mean - state_mean
        exact_state = exact_state + weight * (
            branch.state_factor @ branch.state_factor.T
            + torch.outer(offset, offset)
        )
    torch.testing.assert_close(state_factor @ state_factor.T, exact_state)
    assert belief.maximum_sufficient_rank == 4


def test_information_sandwich_and_single_component_reduction():
    one = RaoBlackwellizedBelief([
        _branch(
            1.0,
            [0.1, -0.3, 0.2],
            [[0.7, 0.0], [0.2, 0.4], [-0.1, 0.3]],
        )
    ])
    noise = torch.tensor([0.4, 0.8, 0.5], dtype=torch.float64)
    bounds = one.information_bounds([0, 1, 2], noise)
    exact = 0.5 * torch.linalg.slogdet(
        torch.eye(2, dtype=torch.float64)
        + one.branches[0].measurement_factor.T
        @ torch.diag(noise.reciprocal())
        @ one.branches[0].measurement_factor
    ).logabsdet.item()
    assert math.isclose(bounds.lower, exact, rel_tol=1e-12)
    assert math.isclose(bounds.upper, exact, rel_tol=1e-12)
    estimate = one.estimate_information(
        [0, 1, 2],
        noise,
        num_samples=8,
        generator=torch.Generator().manual_seed(3),
    )
    assert math.isclose(estimate.total, exact, rel_tol=1e-12)
    assert estimate.mode == 0.0

    mixture = RaoBlackwellizedBelief([
        _branch(0.5, [-2.0, 0.0], [[0.5], [0.2]]),
        _branch(0.5, [2.0, 0.0], [[0.3], [-0.1]]),
    ])
    mixture_bounds = mixture.information_bounds([0, 1], 0.7)
    mixture_estimate = mixture.estimate_information(
        [0, 1],
        0.7,
        num_samples=20_000,
        generator=torch.Generator().manual_seed(9),
    )
    tolerance = 4.0 * mixture_estimate.standard_error + 5e-3
    assert mixture_estimate.total >= mixture_bounds.lower - tolerance
    assert mixture_estimate.total <= mixture_bounds.upper + tolerance
    assert mixture_estimate.mode >= -tolerance
    assert (
        mixture_bounds.upper - mixture_bounds.lower
        <= mixture_bounds.entropy_cap + 1e-12
    )


def test_operator_sufficient_dimension_is_independent_of_action_size():
    dimension = 200
    means = torch.zeros(3, dimension, dtype=torch.float64)
    means[0, 0] = -2.0
    means[1, 0] = 1.0
    means[2, 0] = 3.0
    factors = []
    for index in range(3):
        factor = torch.zeros(dimension, 2, dtype=torch.float64)
        factor[1, 0] = 0.5 + 0.1 * index
        factor[2, 1] = 0.4 - 0.05 * index
        factors.append(factor)
    belief = RaoBlackwellizedBelief([
        DiffusionBeliefBranch(0.2, means[0], factors[0]),
        DiffusionBeliefBranch(0.3, means[1], factors[1]),
        DiffusionBeliefBranch(0.5, means[2], factors[2]),
    ])

    projection = belief.project_action(torch.arange(dimension), 0.25)
    assert projection.dimension == 3
    assert projection.dimension <= belief.maximum_sufficient_rank
    assert belief.maximum_sufficient_rank == 8

    outcomes, labels = projection.sample(
        32, generator=torch.Generator().manual_seed(17)
    )
    assert outcomes.shape == (32, 3)
    assert labels.shape == (32,)
    assert bool(torch.isfinite(projection.mixture_logpdf(outcomes)).all())


def test_projected_conditioning_matches_full_gaussian_mixture_update():
    belief = RaoBlackwellizedBelief([
        _branch(
            0.4,
            [0.2, -0.6, 1.0],
            [[0.8, 0.1], [0.3, -0.2], [0.2, 0.7]],
            state_mean=[0.5, -0.4],
            state_factor=[[0.6, 0.2], [-0.1, 0.8]],
        ),
        _branch(
            0.6,
            [-0.5, 0.4, 0.2],
            [[0.1], [0.5], [0.9]],
            state_mean=[-0.3, 0.7],
            state_factor=[[0.2], [0.9]],
        ),
    ])
    indices = torch.tensor([0, 2])
    noise = torch.tensor([0.3, 0.6], dtype=torch.float64)
    observation = torch.tensor([0.9, -0.2], dtype=torch.float64)
    conditioned = belief.condition(indices, observation, noise)

    direct_log_weights = []
    for original, updated, weight in zip(
        belief.branches, conditioned.branches, belief.weights
    ):
        selected_factor = original.measurement_factor[indices]
        whitened_factor = selected_factor / noise.sqrt().unsqueeze(1)
        whitened_innovation = (
            observation - original.measurement_mean[indices]
        ) / noise.sqrt()
        precision = (
            torch.eye(original.rank, dtype=torch.float64)
            + whitened_factor.T @ whitened_factor
        )
        coefficient_covariance = torch.linalg.inv(precision)
        coefficient_mean = (
            coefficient_covariance
            @ whitened_factor.T
            @ whitened_innovation
        )
        expected_mean = (
            original.measurement_mean
            + original.measurement_factor @ coefficient_mean
        )
        expected_covariance = (
            original.measurement_factor
            @ coefficient_covariance
            @ original.measurement_factor.T
        )
        torch.testing.assert_close(updated.measurement_mean, expected_mean)
        torch.testing.assert_close(
            updated.measurement_factor @ updated.measurement_factor.T,
            expected_covariance,
        )

        covariance = (
            selected_factor @ selected_factor.T + torch.diag(noise)
        )
        residual = observation - original.measurement_mean[indices]
        sign, logdet = torch.linalg.slogdet(covariance)
        assert sign > 0
        log_likelihood = -0.5 * (
            len(indices) * math.log(2 * math.pi)
            + logdet
            + residual @ torch.linalg.solve(covariance, residual)
        )
        direct_log_weights.append(weight.log() + log_likelihood)

    expected_weights = torch.softmax(torch.stack(direct_log_weights), dim=0)
    torch.testing.assert_close(conditioned.weights, expected_weights)


def test_safe_bound_screening_removes_zero_information_action():
    factor = torch.tensor([[2.0], [0.0]], dtype=torch.float64)
    belief = RaoBlackwellizedBelief([
        DiffusionBeliefBranch(
            1.0,
            torch.zeros(2, dtype=torch.float64),
            factor,
        )
    ])
    actions = {0: torch.tensor([0]), 1: torch.tensor([1])}
    survivors, evaluations = screen_actions(belief, actions, 1.0)
    assert survivors == (0,)
    by_action = {row.action: row for row in evaluations}
    assert not by_action[0].safely_pruned
    assert by_action[1].safely_pruned


def test_horizon_two_prefers_mode_router_over_myopic_ternary_action():
    # Six equally likely point modes.  Action 0 has the largest immediate
    # information (three mode pairs).  Action 1 only separates left/right, but
    # then action 2 identifies every left mode and action 3 every right mode.
    # Thus the adaptive two-action tree rooted at 1 identifies all six modes.
    amplitude = 8.0
    action_means = torch.tensor(
        [
            [-2, -1, 0, 0],
            [0, -1, -2, 0],
            [2, -1, 2, 0],
            [-2, 1, 0, 0],
            [0, 1, 0, -2],
            [2, 1, 0, 2],
        ],
        dtype=torch.float64,
    ) * amplitude
    belief = RaoBlackwellizedBelief([
        DiffusionBeliefBranch(
            1.0 / 6.0,
            action_means[index],
            torch.zeros(4, 0, dtype=torch.float64),
            label=f"mode-{index}",
        )
        for index in range(6)
    ])
    actions = {index: torch.tensor([index]) for index in range(4)}
    generator = torch.Generator().manual_seed(1234)

    immediate = {
        action: belief.estimate_information(
            indices,
            0.25,
            num_samples=12_000,
            generator=generator,
        ).total
        for action, indices in actions.items()
    }
    assert max(immediate, key=immediate.get) == 0

    plan = plan_horizon_two(
        belief,
        actions,
        0.25,
        beam_width=4,
        num_fantasies=96,
        information_samples=4_000,
        future_information_samples=2_000,
        generator=torch.Generator().manual_seed(77),
    )
    assert plan.action == 1
    assert plan.expected_future_information > 1.0
    assert plan.diagnostics["maximum_sufficient_rank"] == 5


def test_raosense_mri_runner_acquires_budget_and_counts_tangent_nfe():
    height, width, coils = 4, 5, 2
    dtype = torch.complex64
    truth = torch.complex(
        torch.linspace(-0.5, 0.5, height * width).reshape(height, width),
        torch.zeros(height, width),
    ).to(dtype)
    sensitivities = torch.stack(
        (
            torch.ones(height, width, dtype=dtype),
            (0.7 + 0.2j) * torch.ones(height, width, dtype=dtype),
        )
    )
    full_kspace = fft2c(sensitivities * truth.unsqueeze(0))
    mask = torch.zeros(height, width)
    mask[:, width // 2] = 1.0
    operator = MultiCoilSENSE(
        mask,
        sensitivities,
        torch.full((coils,), 1e-3),
    )

    result = run_raosense_active_mc(
        y_mc=mask * full_kspace,
        sense_op=operator,
        sigma_schedule=torch.tensor([1.0, 0.3, 0.1]),
        denoiser_fn=lambda value, _sigma: value,
        sigma_i_sq_init=torch.full((height, width), 1e-3),
        n_branches=2,
        tangent_rank=1,
        active_lines=1,
        active_rounds=1,
        active_after_frac=0.5,
        active_until_frac=0.5,
        planning_horizon=1,
        information_samples=32,
        planner_candidate_cap=None,
        y_mc_full=full_kspace,
        seed=11,
        return_diagnostics=True,
    )

    assert int((result["mask_final"].sum(dim=0) > 0).sum().item()) == 2
    assert result["nfe"] == 2 * 3 + 2 * 2 * 1
    assert result["diagnostics"]["nfe_base"] == 6
    assert result["diagnostics"]["nfe_tangent"] == 4
    assert len(result["diagnostics"]["acquisitions"]) == 1
    assert bool(torch.isfinite(result["recon"]).all())
    torch.testing.assert_close(
        result["branch_weights"].sum(),
        torch.tensor(1.0),
    )
