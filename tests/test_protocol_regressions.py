import torch

import src.samplers.multicoil_pv as multicoil_pv
from poc_natural.guided_adm import (
    load_heldout_image_folder,
    validate_imagenet_validation_layout,
)
from poc_natural.pvas_core import (
    acquire_blocks,
    init_block_state,
    random_block_scores,
)
from src.models.edm_loader import load_edm_model
from poc_natural.run_cifar import paired_bootstrap_mean_ci
from src.samplers.multicoil_pv import (
    build_active_schedule,
    random_complement_order,
)
from src.samplers.mri_forward import fft2c
from src.samplers.sense import MultiCoilSENSE, estimate_acs_rss_scale
from src.utils.metrics import (
    stacked_nmse_from_sufficient_statistics,
    stacked_psnr_from_sufficient_statistics,
)


def test_natural_static_and_adaptive_random_share_mask_and_measurements():
    x0 = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    init_gen = torch.Generator(device="cpu").manual_seed(19)
    state = init_block_state(
        x0=x0,
        frac=0.125,
        block=4,
        sigma_y=0.1,
        gen=init_gen,
        with_lr=False,
    )
    scores = random_block_scores(1, 4, 4, seed=23, device=x0.device)

    # Up-front acquisition and two-round revelation of the same ranking.
    static = acquire_blocks(
        state, x0, scores, 6,
        torch.Generator(device="cpu").manual_seed(31),
    )
    adaptive = acquire_blocks(
        state, x0, scores, 3,
        torch.Generator(device="cpu").manual_seed(37),
    )
    adaptive = acquire_blocks(
        adaptive, x0, scores, 3,
        torch.Generator(device="cpu").manual_seed(41),
    )

    torch.testing.assert_close(static.mask, adaptive.mask)
    torch.testing.assert_close(static.y, adaptive.y)


def test_active_schedules_preserve_budget_and_unique_steps():
    schedule = build_active_schedule(
        total_steps=20,
        total_atoms=17,
        rounds=4,
        after_frac=0.45,
        until_frac=0.85,
    )
    assert len(schedule) == 4
    assert len(set(schedule)) == 4
    assert sum(schedule.values()) == 17


def test_random_complement_order_never_contains_observed_columns():
    mask = torch.zeros(7, 13)
    mask[:, [1, 4, 9]] = 1
    order = random_complement_order(mask, seed=5)
    assert len(order) == 10
    assert set(order).isdisjoint({1, 4, 9})
    assert len(set(order)) == len(order)


def test_acs_normalization_ignores_unacquired_columns():
    generator = torch.Generator().manual_seed(13)
    real = torch.randn(3, 12, 20, generator=generator)
    imag = torch.randn(3, 12, 20, generator=generator)
    kspace = torch.complex(real, imag)
    center_fraction = 0.2

    perturbed = kspace.clone()
    acs_width = int(round(center_fraction * kspace.shape[-1]))
    center = kspace.shape[-1] // 2
    start = center - acs_width // 2
    stop = center + acs_width // 2
    perturbed[..., :start] += 10_000
    perturbed[..., stop:] -= 10_000j

    torch.testing.assert_close(
        estimate_acs_rss_scale(kspace, center_fraction=center_fraction),
        estimate_acs_rss_scale(perturbed, center_fraction=center_fraction),
    )


def test_edm_loader_never_clones_mutable_source_implicitly(tmp_path):
    missing_checkout = tmp_path / "missing_adps"
    try:
        load_edm_model(
            str(tmp_path / "checkpoint"),
            adps_root=str(missing_checkout),
        )
    except FileNotFoundError as error:
        assert "explicit commit" in str(error)
    else:
        raise AssertionError("missing ADPS checkout was silently accepted")
    assert not missing_checkout.exists()


def test_sensitivity_refresh_recomputes_forward_before_acquisition(monkeypatch):
    height, width, coils = 4, 5, 2
    dtype = torch.complex64
    sensitivity = torch.ones(coils, height, width, dtype=dtype)
    mask = torch.zeros(height, width)
    mask[:, width // 2] = 1.0
    full_kspace = fft2c(
        sensitivity * torch.ones(height, width, dtype=dtype).unsqueeze(0)
    )
    operator = MultiCoilSENSE(
        mask,
        sensitivity,
        torch.full((coils,), 1e-3),
    )

    original_fft = multicoil_pv.fft2c
    forward_inputs = []

    def recording_fft(value):
        forward_inputs.append(value.detach().clone())
        return original_fft(value)

    monkeypatch.setattr(multicoil_pv, "fft2c", recording_fft)
    monkeypatch.setattr(
        multicoil_pv,
        "refine_sens_maps_inloop",
        lambda **kwargs: 2.0 * kwargs["sens_prev"],
    )

    multicoil_pv.run_fakgd_mc_pv(
        y_mc=mask * full_kspace,
        sense_op=operator,
        sigma_schedule=torch.tensor([1.0, 0.1]),
        denoiser_fn=lambda value, sigma: torch.ones_like(value),
        sigma_i_sq_init=torch.full((height, width), 1e-3),
        use_pv_gate=False,
        refine_sens=True,
        sens_refresh_every=1,
        sens_warmup_frac=0.0,
        active_lines=1,
        active_rounds=1,
        active_after_frac=0.0,
        active_until_frac=0.0,
        active_score="random",
        y_mc_full=full_kspace,
        seed=3,
    )

    # Step 0 forward, refreshed forward for immediate acquisition, step 1
    # forward. Before the fix the middle call was absent.
    assert len(forward_inputs) == 3
    torch.testing.assert_close(forward_inputs[1], 2.0 * forward_inputs[0])


def test_imagenet_layout_validation_fails_closed(tmp_path):
    for index in range(1, 4):
        (tmp_path / f"ILSVRC2012_val_{index:08d}.JPEG").touch()
    provenance = validate_imagenet_validation_layout(
        tmp_path, expected_count=3
    )
    assert provenance["canonical_basenames_verified"]

    (tmp_path / "ILSVRC2012_val_00000003.JPEG").rename(
        tmp_path / "renamed.JPEG"
    )
    try:
        validate_imagenet_validation_layout(tmp_path, expected_count=3)
    except ValueError as error:
        assert "not a canonical" in str(error)
    else:
        raise AssertionError("noncanonical ImageNet layout was accepted")


def test_paired_bootstrap_interval_is_deterministic_and_contains_mean():
    differences = [0.1, 0.2, 0.3, 0.4]
    first = paired_bootstrap_mean_ci(
        differences, seed=9, resamples=2_000
    )
    second = paired_bootstrap_mean_ci(
        differences, seed=9, resamples=2_000
    )
    assert first == second
    assert first[0] <= sum(differences) / len(differences) <= first[1]


def test_stacked_volume_metrics_match_fastmri_grain():
    target = torch.tensor([
        [[0.0, 1.0], [2.0, 3.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ])
    prediction = target + torch.tensor([
        [[0.0, 0.5], [-0.5, 0.0]],
        [[0.25, -0.25], [0.0, 0.5]],
    ])
    squared_error_sum = float((target - prediction).square().sum())
    target_squared_sum = float(target.square().sum())

    expected_psnr = float(
        10.0 * torch.log10(
            target.max().square()
            / ((target - prediction).square().mean())
        )
    )
    expected_nmse = squared_error_sum / target_squared_sum

    assert abs(
        stacked_psnr_from_sufficient_statistics(
            squared_error_sum,
            target.numel(),
            float(target.max()),
        )
        - expected_psnr
    ) < 1e-6
    assert stacked_nmse_from_sufficient_statistics(
        squared_error_sum,
        target_squared_sum,
    ) == expected_nmse


def test_imagenet_sample_ids_do_not_depend_on_folder_topology(tmp_path):
    from PIL import Image

    flat = tmp_path / "flat"
    nested = tmp_path / "nested"
    flat.mkdir()
    (nested / "class_a").mkdir(parents=True)
    (nested / "class_b").mkdir(parents=True)
    names = [f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, 5)]
    for index, name in enumerate(names):
        image = Image.new("RGB", (8, 8), color=(index, index, index))
        image.save(flat / name)
        destination = (
            nested / ("class_a" if index % 2 == 0 else "class_b") / name
        )
        image.save(destination)

    _, flat_manifest = load_heldout_image_folder(
        flat, n_images=3, seed=17, image_size=8
    )
    _, nested_manifest = load_heldout_image_folder(
        nested, n_images=3, seed=17, image_size=8
    )
    assert [
        entry["relative_path"].split("/")[-1] for entry in flat_manifest
    ] == [
        entry["relative_path"].split("/")[-1] for entry in nested_manifest
    ]
    assert [
        entry["sha256"] for entry in flat_manifest
    ] == [
        entry["sha256"] for entry in nested_manifest
    ]


def test_imagenet_offset_reserves_disjoint_headline_ids(tmp_path):
    from PIL import Image

    for index in range(1, 11):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(
            tmp_path / f"ILSVRC2012_val_{index:08d}.JPEG"
        )
    _, pilot = load_heldout_image_folder(
        tmp_path,
        n_images=3,
        seed=29,
        sample_offset=0,
        image_size=8,
    )
    _, headline = load_heldout_image_folder(
        tmp_path,
        n_images=5,
        seed=29,
        sample_offset=3,
        image_size=8,
    )
    assert {row["sha256"] for row in pilot}.isdisjoint(
        {row["sha256"] for row in headline}
    )
