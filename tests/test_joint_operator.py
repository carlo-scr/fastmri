import math

import torch

from src.samplers.joint_operator import (
    augment_branch_with_nuisance,
    finite_difference_nuisance_measurement_factor,
    gaussian_target_information,
)
from src.samplers.rao_belief import DiffusionBeliefBranch


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
