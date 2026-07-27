import torch

from src.samplers.mri_forward import fft2c
from src.samplers.multicoil_pv import run_raosense_active_mc
from src.samplers.sense import MultiCoilSENSE


def _tiny_problem():
    generator = torch.Generator().manual_seed(7)
    height, width, coils = 4, 6, 2
    image = torch.complex(
        torch.randn(height, width, generator=generator),
        torch.randn(height, width, generator=generator),
    )
    sensitivity = torch.full(
        (coils, height, width),
        1.0 / coils**0.5,
        dtype=torch.complex64,
    )
    full_kspace = fft2c(sensitivity * image.unsqueeze(0))
    mask = torch.zeros(height, width)
    mask[:, width // 2] = 1.0
    operator = MultiCoilSENSE(
        mask,
        sensitivity,
        torch.full((coils,), 1e-3),
    )
    return full_kspace, mask, operator


def test_raosense_mri_h1_and_h2_preserve_budget_and_nfe():
    full_kspace, initial_mask, operator = _tiny_problem()
    common = dict(
        y_mc=initial_mask * full_kspace,
        sense_op=operator,
        sigma_schedule=torch.tensor([1.0, 0.1]),
        denoiser_fn=lambda value, sigma: value / (1.0 + sigma),
        sigma_i_sq_init=torch.full(initial_mask.shape, 1e-3),
        n_branches=2,
        tangent_rank=1,
        active_lines=1,
        active_rounds=1,
        active_after_frac=0.0,
        active_until_frac=0.0,
        beam_width=2,
        num_fantasies=3,
        information_samples=8,
        planner_candidate_cap=None,
        y_mc_full=full_kspace,
        seed=11,
        return_diagnostics=True,
    )

    greedy = run_raosense_active_mc(planning_horizon=1, **common)
    horizon_two = run_raosense_active_mc(planning_horizon=2, **common)

    expected_columns = int((initial_mask.sum(dim=0) > 0).sum()) + 1
    expected_nfe = 2 * 2 + 2 * 2 * 1 * 1
    for result in (greedy, horizon_two):
        actual_columns = int(
            (result["mask_final"].sum(dim=0) > 0).sum().item()
        )
        assert actual_columns == expected_columns
        assert result["nfe"] == expected_nfe
        assert result["recon"].shape == initial_mask.shape
        assert len(result["diagnostics"]["acquisitions"]) == 1
        assert result["diagnostics"]["nfe_base"] == 4
        assert result["diagnostics"]["nfe_tangent"] == 4

    assert (
        horizon_two["diagnostics"]["acquisitions"][0][
            "expected_future_information"
        ]
        >= 0.0
    )
