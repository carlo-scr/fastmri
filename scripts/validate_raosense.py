"""CPU falsification suite for the RaoSense belief and planner.

The script produces measured evidence for four claims that do not require an
image checkpoint:

1. the Rao--Blackwell covariance estimator has lower empirical MSE than a
   point-particle sample covariance under the same component draws;
2. exact-under-belief information lies inside the analytic sandwich;
3. a thousands-dimensional action compresses to ``K - 1 + sum ranks``;
4. outcome-dependent horizon-two planning solves a mode-routing example that
   defeats the myopic action;
5. pretending an uncertain forward operator is known overstates target-image
   information.

These are analytic/synthetic validation results, not imaging results.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.joint_operator import gaussian_target_information
from src.samplers.rao_belief import (
    DiffusionBeliefBranch,
    RaoBlackwellizedBelief,
    plan_horizon_two,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--covariance_trials", type=int, default=300)
    parser.add_argument("--particles", type=int, default=12)
    parser.add_argument("--action_dimension", type=int, default=4096)
    parser.add_argument("--information_samples", type=int, default=10_000)
    parser.add_argument(
        "--output",
        default="outputs/raosense/analytic_validation.json",
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)

    report = {
        "status": "ANALYTIC_VALIDATION",
        "git_commit": _git_commit(),
        "config": vars(args),
        "rao_blackwell": _validate_rao_blackwell(
            seed=args.seed,
            trials=args.covariance_trials,
            particles=args.particles,
        ),
        "information_and_compression": _validate_information(
            seed=args.seed + 1,
            action_dimension=args.action_dimension,
            samples=args.information_samples,
        ),
        "outcome_routing": _validate_outcome_routing(args.seed + 2),
        "operator_uncertainty": _validate_operator_uncertainty(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"saved {output}")


def _validate_rao_blackwell(
    *,
    seed: int,
    trials: int,
    particles: int,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weights = torch.tensor([0.2, 0.5, 0.3])
    means = torch.tensor(
        [[-1.0, 0.2, 0.5], [0.4, 0.8, -0.3], [1.2, -0.7, 0.1]]
    )
    factors = (
        torch.tensor([[0.6, 0.0], [0.2, 0.3], [-0.1, 0.4]]),
        torch.tensor([[0.2], [0.7], [0.1]]),
        torch.tensor([[0.5, -0.2], [0.0, 0.4], [0.3, 0.2]]),
    )
    mean = (weights.unsqueeze(1) * means).sum(dim=0)
    exact = torch.zeros(3, 3)
    covariances = []
    for component in range(3):
        covariance = factors[component] @ factors[component].T
        covariances.append(covariance)
        offset = means[component] - mean
        exact = exact + weights[component] * (
            covariance + torch.outer(offset, offset)
        )

    naive_errors = []
    rao_errors = []
    for _ in range(trials):
        labels = torch.multinomial(
            weights,
            particles,
            replacement=True,
            generator=generator,
        )
        samples = []
        conditional_covariance = torch.zeros(3, 3)
        conditional_means = []
        for label_tensor in labels:
            label = int(label_tensor.item())
            coefficient = torch.randn(
                factors[label].shape[1], generator=generator
            )
            samples.append(
                means[label] + factors[label] @ coefficient
            )
            conditional_covariance = (
                conditional_covariance + covariances[label] / particles
            )
            conditional_means.append(means[label])
        samples = torch.stack(samples)
        conditional_means = torch.stack(conditional_means)
        naive = torch.cov(samples.T, correction=1)
        rao = (
            conditional_covariance
            + torch.cov(conditional_means.T, correction=1)
        )
        naive_errors.append((naive - exact).square().sum().item())
        rao_errors.append((rao - exact).square().sum().item())
    naive_mse = sum(naive_errors) / len(naive_errors)
    rao_mse = sum(rao_errors) / len(rao_errors)
    return {
        "particle_covariance_mse": naive_mse,
        "rao_blackwell_covariance_mse": rao_mse,
        "relative_mse": rao_mse / naive_mse,
        "mse_reduction_percent": 100.0 * (1.0 - rao_mse / naive_mse),
        "trials": trials,
        "particles": particles,
    }


def _validate_information(
    *,
    seed: int,
    action_dimension: int,
    samples: int,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    branches = []
    branch_count = 4
    rank = 2
    shared_basis = torch.randn(
        action_dimension,
        branch_count * rank + branch_count - 1,
        generator=generator,
    ) / math.sqrt(action_dimension)
    for component in range(branch_count):
        mean_coefficients = torch.randn(
            branch_count - 1, generator=generator
        )
        factor_coefficients = torch.randn(
            branch_count * rank + branch_count - 1,
            rank,
            generator=generator,
        )
        mean = shared_basis[:, : branch_count - 1] @ mean_coefficients
        factor = shared_basis @ factor_coefficients
        branches.append(
            DiffusionBeliefBranch(
                weight=1.0 / branch_count,
                measurement_mean=mean,
                measurement_factor=factor,
            )
        )
    belief = RaoBlackwellizedBelief(branches)
    indices = torch.arange(action_dimension)
    bounds = belief.information_bounds(indices, 0.5)
    estimate = belief.estimate_information(
        indices,
        0.5,
        num_samples=samples,
        generator=generator,
    )
    projection = belief.project_action(indices, 0.5)
    return {
        "scalar_measurements": action_dimension,
        "theoretical_rank_cap": belief.maximum_sufficient_rank,
        "measured_sufficient_rank": projection.dimension,
        "compression_ratio": action_dimension / max(1, projection.dimension),
        "information_lower": bounds.lower,
        "information_estimate": estimate.total,
        "information_standard_error": estimate.standard_error,
        "information_upper": bounds.upper,
        "moment_gaussian_upper": bounds.gaussian_upper,
        "inside_sandwich": (
            estimate.total
            >= bounds.lower - 4.0 * estimate.standard_error
            and estimate.total
            <= bounds.upper + 4.0 * estimate.standard_error
        ),
    }


def _validate_outcome_routing(seed: int) -> dict:
    amplitude = 8.0
    means = torch.tensor(
        [
            [-2, -1, 0, 0],
            [0, -1, -2, 0],
            [2, -1, 2, 0],
            [-2, 1, 0, 0],
            [0, 1, 0, -2],
            [2, 1, 0, 2],
        ]
    ) * amplitude
    belief = RaoBlackwellizedBelief([
        DiffusionBeliefBranch(
            weight=1.0 / 6.0,
            measurement_mean=means[index],
            measurement_factor=torch.zeros(4, 0),
            label=f"mode-{index}",
        )
        for index in range(6)
    ])
    actions = {index: torch.tensor([index]) for index in range(4)}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    immediate = {
        action: belief.estimate_information(
            indices,
            0.25,
            num_samples=12_000,
            generator=generator,
        ).total
        for action, indices in actions.items()
    }
    plan = plan_horizon_two(
        belief,
        actions,
        0.25,
        beam_width=4,
        num_fantasies=128,
        information_samples=4_000,
        future_information_samples=2_000,
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
    )
    return {
        "myopic_action": max(immediate, key=immediate.get),
        "horizon_two_action": plan.action,
        "immediate_information_by_action": immediate,
        "selected_immediate_information": plan.immediate_information,
        "selected_expected_future_information": (
            plan.expected_future_information
        ),
        "root_changed": plan.action != max(immediate, key=immediate.get),
    }


def _validate_operator_uncertainty() -> dict:
    image_response = 2.0
    nuisance_response = 3.0
    noise = 0.5
    branch = DiffusionBeliefBranch(
        weight=1.0,
        measurement_mean=torch.zeros(1),
        measurement_factor=torch.tensor(
            [[image_response, nuisance_response]]
        ),
        state_mean=torch.zeros(2),
        state_factor=torch.eye(2),
    )
    diagnostic = gaussian_target_information(
        branch,
        indices=[0],
        noise_variance=noise,
        target_state_rows=[0],
        nuisance_state_rows=[1],
    )
    return {
        "joint_information": diagnostic.joint_information,
        "target_information_with_unknown_operator": (
            diagnostic.target_information
        ),
        "target_information_if_operator_known": (
            diagnostic.known_nuisance_information
        ),
        "known_operator_optimism": diagnostic.known_operator_optimism,
        "optimism_ratio": (
            diagnostic.known_nuisance_information
            / diagnostic.target_information
        ),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
