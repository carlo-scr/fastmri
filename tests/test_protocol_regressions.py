import torch

from poc_natural.pvas_core import (
    acquire_blocks,
    init_block_state,
    random_block_scores,
)
from src.samplers.multicoil_pv import (
    build_active_schedule,
    random_complement_order,
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
