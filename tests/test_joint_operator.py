import math

import torch

from src.samplers.joint_operator import (
    augment_branch_with_nuisance,
    estimate_mixture_target_information,
    finite_difference_nuisance_measurement_factor,
    gaussian_target_information,
    mixture_target_information_bounds,
    screen_target_information_actions,
)
from src.samplers.rao_belief import (
    DiffusionBeliefBranch,
    RaoBlackwellizedBelief,
)


def test_nuisance_augmentation_builds_aligned_block_factor():
    image_branch = DiffusionBeliefBranch(
        weight=1.0,
        measurement_mean=torch.tensor([0.2, -0.1], dtype=torch.float64),
        measurement_factor=torch.tensor(
            [[1.0], [0.5]], dtype=torch.float64
        ),
        state_mean=torch.tensor([0.3, 0.7, -0.2], dtype=torch.float64),
        state_factor=torch.tensor(
            [[0.8], [0.1], [-0.4]], dtype=torch.float64
        ),
    )
    nuisance_factor = torch.tensor(
        [[0.2, 0.0], [0.1, 0.3]], dtype=torch.float64
    )
    nuisance_measurement_factor = torch.tensor(
        [[0.4, -0.1], [0.2, 0.6]], dtype=torch.float64
    )
    joint = augment_branch_with_nuisance(
        image_branch,
        nuisance_mean=torch.tensor([0.0, 1.0], dtype=torch.float64),
        nuisance_factor=nuisance_factor,
        nuisance_measurement_factor=nuisance_measurement_factor,
    )

    assert joint.rank == 3
    torch.testing.assert_close(
        joint.measurement_factor,
        torch.cat(
            (
                image_branch.measurement_factor,
                nuisance_measurement_factor,
            ),
            dim=1,
        ),
    )
    assert joint.state_factor is not None
    expected = torch.zeros(5, 3, dtype=torch.float64)
    expected[:3, :1] = image_branch.state_factor
    expected[3:, 1:] = nuisance_factor
    torch.testing.assert_close(joint.state_factor, expected)


def test_target_information_exposes_known_operator_optimism():
    # x, phi ~ N(0, 1), y = a*x + b*phi + noise.
    image_response = 2.0
    nuisance_response = 3.0
    noise = 0.5
    branch = DiffusionBeliefBranch(
        weight=1.0,
        measurement_mean=torch.zeros(1, dtype=torch.float64),
        measurement_factor=torch.tensor(
            [[image_response, nuisance_response]], dtype=torch.float64
        ),
        state_mean=torch.zeros(2, dtype=torch.float64),
        state_factor=torch.eye(2, dtype=torch.float64),
    )
    diagnostic = gaussian_target_information(
        branch,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
        nuisance_state_rows=[1],
    )

    expected_joint = 0.5 * math.log(
        1.0 + (image_response**2 + nuisance_response**2) / noise
    )
    expected_target = 0.5 * math.log(
        1.0 + image_response**2 / (noise + nuisance_response**2)
    )
    expected_known = 0.5 * math.log(
        1.0 + image_response**2 / noise
    )
    assert math.isclose(
        diagnostic.joint_information, expected_joint, rel_tol=1e-12
    )
    assert math.isclose(
        diagnostic.target_information, expected_target, rel_tol=1e-12
    )
    assert math.isclose(
        diagnostic.known_nuisance_information,
        expected_known,
        rel_tol=1e-12,
    )
    assert diagnostic.known_operator_optimism > 0
    assert diagnostic.target_rank == 1
    assert diagnostic.nuisance_rank == 1


def test_mixture_target_information_reduces_to_single_branch_analytic():
    # x, phi ~ N(0, 1).  The first action observes both variables, while the
    # second is nuisance-only and therefore has zero information about x.
    image_response = 2.0
    nuisance_response = 3.0
    nuisance_only_response = 4.0
    noise = 0.5
    belief = RaoBlackwellizedBelief([
        DiffusionBeliefBranch(
            weight=1.0,
            measurement_mean=torch.zeros(2, dtype=torch.float64),
            measurement_factor=torch.tensor(
                [
                    [image_response, nuisance_response],
                    [0.0, nuisance_only_response],
                ],
                dtype=torch.float64,
            ),
            state_mean=torch.zeros(2, dtype=torch.float64),
            state_factor=torch.eye(2, dtype=torch.float64),
        )
    ])
    expected = 0.5 * math.log(
        1.0 + image_response**2 / (noise + nuisance_response**2)
    )

    bounds = mixture_target_information_bounds(
        belief,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
    )
    estimate = estimate_mixture_target_information(
        belief,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
        num_samples=8,
        generator=torch.Generator().manual_seed(5),
    )
    assert math.isclose(bounds.lower, expected, rel_tol=1e-12)
    assert math.isclose(bounds.upper, expected, rel_tol=1e-12)
    assert bounds.gaussian_upper > bounds.upper
    assert math.isclose(estimate.total, expected, rel_tol=1e-12)
    assert math.isclose(estimate.within_target, expected, rel_tol=1e-12)
    assert estimate.mode == 0.0

    survivors, evaluations = screen_target_information_actions(
        belief,
        actions={0: [0], 1: [1]},
        noise_variance=noise,
        target_state_rows=[0],
    )
    assert survivors == (0,)
    by_action = {row.action: row for row in evaluations}
    assert not by_action[0].safely_pruned
    assert by_action[1].safely_pruned
    assert by_action[1].bounds.upper == 0.0


def test_mixture_target_information_adds_two_mode_information():
    # Both branches have the same within-branch image/nuisance channel, but
    # well-separated predictive means.  The action therefore supplies the
    # analytic target information plus almost all categorical entropy.
    weights = (0.35, 0.65)
    means = (-8.0, 8.0)
    image_response = 1.2
    nuisance_response = 0.7
    noise = 0.4
    branches = []
    for index, (weight, mean) in enumerate(zip(weights, means)):
        branches.append(
            DiffusionBeliefBranch(
                weight=weight,
                measurement_mean=torch.tensor(
                    [mean], dtype=torch.float64
                ),
                measurement_factor=torch.tensor(
                    [[image_response, nuisance_response]],
                    dtype=torch.float64,
                ),
                state_mean=torch.tensor(
                    [float(index), 0.0], dtype=torch.float64
                ),
                state_factor=torch.eye(2, dtype=torch.float64),
                label=f"mode-{index}",
            )
        )
    belief = RaoBlackwellizedBelief(branches)
    expected_within = 0.5 * math.log(
        1.0 + image_response**2 / (noise + nuisance_response**2)
    )
    entropy = -sum(weight * math.log(weight) for weight in weights)

    bounds = mixture_target_information_bounds(
        belief,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
    )
    estimate = estimate_mixture_target_information(
        belief,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
        num_samples=20_000,
        generator=torch.Generator().manual_seed(23),
    )
    tolerance = 5.0 * estimate.standard_error + 5e-3

    assert math.isclose(bounds.lower, expected_within, rel_tol=1e-12)
    assert math.isclose(
        bounds.upper,
        expected_within + entropy,
        rel_tol=1e-12,
    )
    assert math.isclose(
        estimate.within_target, expected_within, rel_tol=1e-12
    )
    assert abs(estimate.mode - entropy) <= tolerance
    assert estimate.total >= bounds.lower - tolerance
    assert estimate.total <= bounds.upper + tolerance


def test_nuisance_forward_factor_matches_analytic_jacobian():
    state = torch.tensor([0.7, -0.2], dtype=torch.float64)
    nuisance = torch.tensor([0.3, -0.4], dtype=torch.float64)
    factor = torch.tensor(
        [[0.2, 0.0], [0.1, 0.5]], dtype=torch.float64
    )

    def forward(image, parameters):
        p0, p1 = parameters
        return torch.stack(
            (
                (1.0 + p0) * image[0] + p1 * image[1],
                p0.square() + image[1],
            )
        )

    estimate = finite_difference_nuisance_measurement_factor(
        state_mean=state,
        nuisance_mean=nuisance,
        nuisance_factor=factor,
        forward=forward,
        fd_step=1e-4,
    )
    jacobian = torch.tensor(
        [[state[0], state[1]], [2.0 * nuisance[0], 0.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        estimate,
        jacobian @ factor,
        rtol=1e-9,
        atol=1e-9,
    )
