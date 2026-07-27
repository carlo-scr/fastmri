"""RaoSense adapter for R2DM range-image diffusion.

The adapter intentionally depends only on the public interface of the
continuous-time diffusion object released with R2DM.  It does not vendor or
modify R2DM.  The expected object provides ``model``, ``log_snr``,
``objective``, ``clip_sample``, and ``clip_sample_range``.

KITTI-360 scans are represented as normalized two-channel (range,
reflectance) images.  A candidate action reveals an azimuth--elevation tile,
so the measurement operator is coordinate selection in the equirectangular
range image.  This gives a local sensing geometry that is deliberately unlike
the global complex SENSE operator used by the MRI experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from .operator_sketch import (
    real_unflatten,
    sketch_operator_posterior_covariance,
)
from .rao_belief import (
    RaoBlackwellizedBelief,
    branch_from_operator_sketch,
    plan_horizon_two,
)


@dataclass(frozen=True)
class LidarAcquisition:
    step: int
    within_round: int
    action: int
    new_scalar_measurements: int
    immediate_information: float
    expected_future_information: float
    effective_sample_size: float
    candidate_cap_pruned: int


def build_lidar_tile_actions(
    *,
    channels: int,
    height: int,
    width: int,
    tile_height: int = 2,
    tile_width: int = 32,
    observed_mask: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> dict[int, torch.Tensor]:
    """Partition a range image into equal physical beam tiles."""
    if min(channels, height, width, tile_height, tile_width) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    if height % tile_height or width % tile_width:
        raise ValueError("tile dimensions must divide the range image")
    if observed_mask is not None:
        if observed_mask.shape != (channels, height, width):
            raise ValueError("observed_mask has the wrong shape")
        observed = observed_mask.to(device=device).bool().reshape(-1)
    else:
        observed = torch.zeros(
            channels * height * width,
            dtype=torch.bool,
            device=device,
        )
    actions = {}
    action = 0
    plane = height * width
    for row_start in range(0, height, tile_height):
        rows = torch.arange(
            row_start,
            row_start + tile_height,
            device=observed.device,
        )
        for column_start in range(0, width, tile_width):
            columns = torch.arange(
                column_start,
                column_start + tile_width,
                device=observed.device,
            )
            pixel = (
                rows[:, None] * width + columns[None, :]
            ).reshape(-1)
            indices = torch.cat(
                [pixel + channel * plane for channel in range(channels)]
            ).to(torch.long)
            indices = indices[~observed[indices]]
            if indices.numel():
                actions[action] = indices
            action += 1
    return actions


def make_lidar_scout_mask(
    *,
    channels: int,
    height: int,
    width: int,
    elevation_stride: int = 4,
    azimuth_stride: int = 4,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return a fixed coarse angular lattice shared by every method."""
    if elevation_stride <= 0 or azimuth_stride <= 0:
        raise ValueError("scout strides must be positive")
    mask = torch.zeros(
        channels, height, width, dtype=torch.bool, device=device
    )
    mask[:, ::elevation_stride, ::azimuth_stride] = True
    return mask


def continuous_x0_prediction(
    diffusion,
    x_t: torch.Tensor,
    step_t: torch.Tensor,
) -> torch.Tensor:
    """Reproduce R2DM's public continuous-time clean prediction."""
    batched, squeezed = _ensure_batch(x_t)
    steps = _step_batch(step_t, batched.shape[0], batched.device)
    log_snr = diffusion.log_snr(steps)
    alpha = log_snr.sigmoid().sqrt()
    sigma = (-log_snr).sigmoid().sqrt()
    prediction = diffusion.model(
        batched, log_snr[:, 0, 0, 0]
    )
    if diffusion.objective == "eps":
        x_0 = (batched - sigma * prediction) / alpha
    elif diffusion.objective == "v":
        x_0 = alpha * batched - sigma * prediction
    elif diffusion.objective == "x_0":
        x_0 = prediction
    else:
        raise ValueError(f"unsupported diffusion objective {diffusion.objective}")
    if diffusion.clip_sample:
        x_0 = x_0.clamp(
            -float(diffusion.clip_sample_range),
            float(diffusion.clip_sample_range),
        )
    return x_0[0] if squeezed else x_0


def continuous_tweedie_coefficient(
    diffusion,
    step_t: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    """Return ``sigma_t^2 / alpha_t`` for the VP Tweedie Jacobian."""
    step = _step_batch(step_t, 1, device)
    log_snr = diffusion.log_snr(step)
    alpha = log_snr.sigmoid().sqrt()
    sigma = (-log_snr).sigmoid().sqrt()
    return float((sigma.square() / alpha).reshape(-1)[0].item())


def continuous_ddim_step_with_x0(
    diffusion,
    x_t: torch.Tensor,
    x_0: torch.Tensor,
    step_t: torch.Tensor,
    step_s: torch.Tensor,
) -> torch.Tensor:
    """Deterministic R2DM DDIM transition using a corrected clean mean."""
    batched, squeezed = _ensure_batch(x_t)
    x0_batched, _ = _ensure_batch(x_0)
    t = _step_batch(step_t, batched.shape[0], batched.device)
    s = _step_batch(step_s, batched.shape[0], batched.device)
    log_snr_t = diffusion.log_snr(t)
    log_snr_s = diffusion.log_snr(s)
    alpha_t = log_snr_t.sigmoid().sqrt()
    sigma_t = (-log_snr_t).sigmoid().sqrt()
    alpha_s = log_snr_s.sigmoid().sqrt()
    sigma_s = (-log_snr_s).sigmoid().sqrt()
    epsilon = (batched - alpha_t * x0_batched) / sigma_t
    x_s = alpha_s * x0_batched + sigma_s * epsilon
    return x_s[0] if squeezed else x_s


@torch.no_grad()
def run_raosense_lidar(
    *,
    diffusion,
    full_scan: torch.Tensor,
    observed_scan: torch.Tensor | None = None,
    initial_mask: torch.Tensor,
    candidate_actions: Mapping[int, Sequence[int] | torch.Tensor] | None = None,
    num_steps: int = 32,
    n_branches: int = 8,
    tangent_rank: int = 2,
    active_actions: int = 8,
    active_rounds: int = 4,
    active_after_frac: float = 0.30,
    active_until_frac: float = 0.75,
    planning_horizon: int = 2,
    beam_width: int = 8,
    num_fantasies: int = 16,
    information_samples: int = 128,
    planner_candidate_cap: int | None = 48,
    measurement_noise_variance: float | torch.Tensor = 1e-3,
    tile_height: int = 2,
    tile_width: int = 32,
    sketch_fd_step: float = 5e-2,
    use_belief_correction: bool = True,
    seed: int = 0,
) -> dict:
    """Retrospective programmable-LiDAR acquisition with an R2DM prior.

    ``full_scan`` is the hidden evaluation reference.  By default the
    retrospective emulator reveals its exact encoded values.  Robustness
    studies must instead pass one once-sampled ``observed_scan``; that tensor,
    never hidden truth, is then used for belief conditioning and every
    data-consistency clamp.
    """
    if full_scan.dim() != 3 or torch.is_complex(full_scan):
        raise ValueError("full_scan must have real shape (channels, H, W)")
    if observed_scan is None:
        observations = full_scan
    else:
        if (
            observed_scan.shape != full_scan.shape
            or observed_scan.dtype != full_scan.dtype
            or observed_scan.device != full_scan.device
            or torch.is_complex(observed_scan)
        ):
            raise ValueError("observed_scan must match full_scan")
        if not bool(torch.isfinite(observed_scan).all()):
            raise ValueError("observed_scan must be finite")
        observations = observed_scan
    if initial_mask.shape != full_scan.shape:
        raise ValueError("initial_mask must match full_scan")
    if num_steps <= 0 or n_branches <= 0 or tangent_rank <= 0:
        raise ValueError("steps, branches, and tangent rank must be positive")
    if planning_horizon not in {1, 2}:
        raise ValueError("planning_horizon must be one or two")

    channels, height, width = full_scan.shape
    device = full_scan.device
    mask = initial_mask.bool().clone()
    if candidate_actions is None:
        action_bank = build_lidar_tile_actions(
            channels=channels,
            height=height,
            width=width,
            tile_height=tile_height,
            tile_width=tile_width,
            observed_mask=mask,
            device=device,
        )
    else:
        action_bank = {
            int(action): torch.as_tensor(
                indices, device=device, dtype=torch.long
            ).flatten()
            for action, indices in candidate_actions.items()
        }
    action_bank = {
        action: indices
        for action, indices in action_bank.items()
        if indices.numel()
    }
    if not action_bank and active_actions:
        raise ValueError("no unobserved candidate actions are available")

    active_schedule = _active_schedule(
        total_steps=num_steps,
        total_actions=active_actions,
        rounds=active_rounds,
        after_frac=active_after_frac,
        until_frac=active_until_frac,
    )
    steps = torch.linspace(
        1.0, 0.0, num_steps + 1, device=device
    )
    branch_generators = [
        torch.Generator(device=device).manual_seed(
            int(seed) + 104_729 * (index + 1)
        )
        for index in range(n_branches)
    ]
    states = [
        torch.randn(
            full_scan.shape,
            generator=branch_generators[index],
            device=device,
            dtype=full_scan.dtype,
        )
        for index in range(n_branches)
    ]
    weights = torch.full(
        (n_branches,),
        1.0 / n_branches,
        device=device,
        dtype=full_scan.dtype,
    )
    observation_vector = observations.reshape(-1)
    nfe = 0
    acquisitions: list[LidarAcquisition] = []

    for step in range(num_steps):
        step_t = steps[step]
        step_s = steps[step + 1]
        means = [
            continuous_x0_prediction(diffusion, state, step_t)
            for state in states
        ]
        nfe += n_branches

        if step in active_schedule:
            coefficient = continuous_tweedie_coefficient(
                diffusion, step_t, device=device
            )
            scale = math_sqrt_positive(coefficient)
            branches = []
            for branch_index in range(n_branches):
                def denoiser(value, _scale):
                    return continuous_x0_prediction(
                        diffusion, value, step_t
                    )

                sketch = sketch_operator_posterior_covariance(
                    x_t=states[branch_index],
                    mu=means[branch_index],
                    sigma_t=scale,
                    denoiser=denoiser,
                    forward=lambda value: value,
                    adjoint=lambda value: value,
                    measurement_template=means[branch_index],
                    num_probes=tangent_rank,
                    fd_step=sketch_fd_step,
                    centered=True,
                    generator=torch.Generator(device="cpu").manual_seed(
                        int(seed)
                        + 1_000_003 * (branch_index + 1)
                        + 7_919 * step
                    ),
                )
                nfe += 2 * tangent_rank
                branches.append(
                    branch_from_operator_sketch(
                        weight=float(weights[branch_index].item()),
                        measurement_mean=means[branch_index],
                        state_mean=means[branch_index],
                        sketch=sketch,
                        label=f"branch-{branch_index}",
                    )
                )
            belief = RaoBlackwellizedBelief(branches)

            for within_round in range(active_schedule[step]):
                feasible = {
                    action: indices[~mask.reshape(-1)[indices]]
                    for action, indices in action_bank.items()
                    if bool((~mask.reshape(-1)[indices]).any())
                }
                if not feasible:
                    break
                cap_pruned = 0
                if (
                    planner_candidate_cap is not None
                    and len(feasible) > planner_candidate_cap
                ):
                    # Engineering cap only: an immediate-information upper
                    # bound is not a safe H=2 root bound.
                    ranked = sorted(
                        feasible,
                        key=lambda action: (
                            belief.information_bounds(
                                feasible[action],
                                measurement_noise_variance,
                            ).upper,
                            -action,
                        ),
                        reverse=True,
                    )
                    keep = set(ranked[:planner_candidate_cap])
                    cap_pruned = len(feasible) - len(keep)
                    feasible = {
                        action: indices
                        for action, indices in feasible.items()
                        if action in keep
                    }
                planner_generator = torch.Generator(
                    device="cpu"
                ).manual_seed(
                    int(seed) + 31_337 * (step + 1) + within_round
                )
                # H1 is the discount-zero intervention through the same
                # all-root estimator used by H2.  With a shared seed, both
                # horizons receive identical immediate-information draws.
                plan = plan_horizon_two(
                    belief,
                    feasible,
                    measurement_noise_variance,
                    beam_width=(
                        min(beam_width, len(feasible))
                        if planning_horizon == 2
                        else len(feasible)
                    ),
                    num_fantasies=(
                        num_fantasies if planning_horizon == 2 else 1
                    ),
                    information_samples=information_samples,
                    discount=1.0 if planning_horizon == 2 else 0.0,
                    generator=planner_generator,
                )
                action = int(plan.action)
                immediate = float(plan.immediate_information)
                future = float(plan.expected_future_information)

                indices = feasible[action]
                belief = belief.condition(
                    indices,
                    observation_vector[indices],
                    measurement_noise_variance,
                )
                weights = belief.weights
                if use_belief_correction:
                    for branch_index, branch in enumerate(belief.branches):
                        assert branch.state_mean is not None
                        means[branch_index] = real_unflatten(
                            branch.state_mean,
                            means[branch_index],
                        )
                mask.reshape(-1)[indices] = True
                acquisitions.append(
                    LidarAcquisition(
                        step=int(step),
                        within_round=int(within_round),
                        action=action,
                        new_scalar_measurements=int(indices.numel()),
                        immediate_information=immediate,
                        expected_future_information=future,
                        effective_sample_size=float(
                            weights.square().sum().reciprocal().item()
                        ),
                        candidate_cap_pruned=int(cap_pruned),
                    )
                )

        for branch_index in range(n_branches):
            corrected_mean = torch.where(
                mask, observations, means[branch_index]
            )
            state_s = continuous_ddim_step_with_x0(
                diffusion,
                states[branch_index],
                corrected_mean,
                step_t,
                step_s,
            )
            log_snr_s = diffusion.log_snr(
                _step_batch(step_s, 1, device)
            )
            alpha_s = log_snr_s.sigmoid().sqrt()[0]
            sigma_s = (-log_snr_s).sigmoid().sqrt()[0]
            known_noise = torch.randn(
                full_scan.shape,
                generator=branch_generators[branch_index],
                device=device,
                dtype=full_scan.dtype,
            )
            known_s = alpha_s * observations + sigma_s * known_noise
            states[branch_index] = torch.where(mask, known_s, state_s)

    states = [torch.where(mask, observations, state) for state in states]
    stacked = torch.stack(states)
    recon = (
        weights.view(n_branches, 1, 1, 1) * stacked
    ).sum(dim=0)
    expected_actions = min(active_actions, len(action_bank))
    if len(acquisitions) != expected_actions:
        raise AssertionError(
            f"expected {expected_actions} actions, got {len(acquisitions)}"
        )
    return {
        "recon": recon,
        "mask_final": mask,
        "branch_recons": states,
        "branch_weights": weights,
        "nfe": nfe,
        "acquisitions": acquisitions,
        "planner_candidate_cap": planner_candidate_cap,
    }


def _ensure_batch(value: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if value.dim() == 3:
        return value.unsqueeze(0), True
    if value.dim() == 4:
        return value, False
    raise ValueError("diffusion state must have shape (C,H,W) or (B,C,H,W)")


def _step_batch(
    value: torch.Tensor | float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    step = torch.as_tensor(value, dtype=torch.float32, device=device)
    if step.numel() == 1:
        return step.reshape(1).expand(batch_size)
    if step.dim() == 1 and step.numel() == batch_size:
        return step
    raise ValueError("diffusion step has incompatible batch size")


def _active_schedule(
    *,
    total_steps: int,
    total_actions: int,
    rounds: int,
    after_frac: float,
    until_frac: float,
) -> dict[int, int]:
    if total_actions <= 0:
        return {}
    rounds = max(1, int(rounds))
    lo = max(0, min(total_steps - 1, round(after_frac * (total_steps - 1))))
    hi = max(lo, min(total_steps - 1, round(until_frac * (total_steps - 1))))
    candidates = list(range(lo, hi + 1))
    if rounds > len(candidates):
        raise ValueError("acquisition rounds do not fit the step window")
    if rounds == 1:
        steps = [lo]
    else:
        steps = [
            candidates[
                round(index * (len(candidates) - 1) / (rounds - 1))
            ]
            for index in range(rounds)
        ]
    base, extra = divmod(total_actions, rounds)
    return {
        step: base + (1 if index < extra else 0)
        for index, step in enumerate(steps)
    }


def math_sqrt_positive(value: float) -> float:
    if not torch.isfinite(torch.tensor(value)) or value <= 0:
        raise ValueError("Tweedie coefficient must be finite and positive")
    return float(value**0.5)
