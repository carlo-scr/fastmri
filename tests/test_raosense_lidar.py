import torch

from src.samplers.raosense_lidar import (
    build_lidar_tile_actions,
    continuous_ddim_step_with_x0,
    continuous_x0_prediction,
    make_lidar_scout_mask,
    run_raosense_lidar,
)


class _ToyModel:
    def __call__(self, value, _condition):
        return 0.1 * value


class _ToyContinuousDiffusion:
    objective = "eps"
    clip_sample = True
    clip_sample_range = 1.0
    model = _ToyModel()

    def log_snr(self, steps):
        ratio = (1.01 - steps) / (steps + 0.01)
        return ratio.log()[:, None, None, None]


def test_lidar_tile_actions_cover_each_unobserved_coordinate_once():
    mask = make_lidar_scout_mask(
        channels=2,
        height=4,
        width=8,
        elevation_stride=2,
        azimuth_stride=2,
    )
    actions = build_lidar_tile_actions(
        channels=2,
        height=4,
        width=8,
        tile_height=2,
        tile_width=4,
        observed_mask=mask,
    )
    coordinates = torch.cat(list(actions.values()))
    unobserved = torch.nonzero(~mask.reshape(-1), as_tuple=False).flatten()
    assert torch.unique(coordinates).numel() == coordinates.numel()
    assert set(coordinates.tolist()) == set(unobserved.tolist())


def test_continuous_r2dm_adapter_uses_correct_vp_equations():
    diffusion = _ToyContinuousDiffusion()
    state = torch.full((2, 3, 4), 0.4)
    step_t = torch.tensor(0.5)
    log_snr = diffusion.log_snr(torch.tensor([0.5]))
    alpha = log_snr.sigmoid().sqrt()[0]
    sigma = (-log_snr).sigmoid().sqrt()[0]
    expected = (state - sigma * 0.1 * state) / alpha
    expected = expected.clamp(-1, 1)
    prediction = continuous_x0_prediction(diffusion, state, step_t)
    torch.testing.assert_close(prediction, expected)

    next_state = continuous_ddim_step_with_x0(
        diffusion,
        state,
        prediction,
        step_t,
        torch.tensor(0.25),
    )
    assert next_state.shape == state.shape
    assert bool(torch.isfinite(next_state).all())


def test_raosense_lidar_runner_acquires_tile_and_counts_nfe():
    diffusion = _ToyContinuousDiffusion()
    full_scan = torch.linspace(-0.8, 0.8, 2 * 4 * 8).reshape(2, 4, 8)
    initial_mask = make_lidar_scout_mask(
        channels=2,
        height=4,
        width=8,
        elevation_stride=2,
        azimuth_stride=2,
    )
    initial_count = int(initial_mask.sum().item())

    result = run_raosense_lidar(
        diffusion=diffusion,
        full_scan=full_scan,
        initial_mask=initial_mask,
        num_steps=3,
        n_branches=2,
        tangent_rank=1,
        active_actions=1,
        active_rounds=1,
        active_after_frac=0.5,
        active_until_frac=0.5,
        planning_horizon=1,
        information_samples=24,
        planner_candidate_cap=None,
        measurement_noise_variance=1e-2,
        tile_height=2,
        tile_width=4,
        seed=5,
    )

    assert result["nfe"] == 2 * 3 + 2 * 2 * 1
    assert len(result["acquisitions"]) == 1
    assert int(result["mask_final"].sum().item()) > initial_count
    assert bool(torch.isfinite(result["recon"]).all())
    torch.testing.assert_close(
        result["branch_weights"].sum(),
        torch.tensor(1.0),
    )
