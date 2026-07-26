"""Analytic falsification suite for operator-space SketchSense.

This script is intentionally independent of image checkpoints and datasets.
It writes measured approximation/selection errors for a non-idempotent
Gaussian covariance, including the legacy squared-row-energy failure.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.operator_sketch import (
    build_atom_information,
    greedy_logdet_select,
    sketch_operator_posterior_covariance,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dim", type=int, default=64)
    parser.add_argument("--measurement_dim", type=int, default=48)
    parser.add_argument("--latent_rank", type=int, default=12)
    parser.add_argument("--probes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16])
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--noise_variance", type=float, default=0.05)
    parser.add_argument("--fd_step", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default="outputs/sketchsense/analytic_validation.json",
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        dtype = torch.float64
        basis = torch.randn(
            args.image_dim, args.latent_rank,
            generator=generator, dtype=dtype,
        )
        eigenvalues = torch.logspace(
            0.4, -0.8, args.latent_rank, dtype=dtype
        )
        jacobian = (
            basis @ torch.diag(eigenvalues) @ basis.T / args.image_dim
            + 0.03 * torch.eye(args.image_dim, dtype=dtype)
        )
        operator = torch.randn(
            args.measurement_dim, args.image_dim,
            generator=generator, dtype=dtype,
        ) / math.sqrt(args.image_dim)
        sigma = float(args.sigma)
        exact = operator @ (sigma**2 * jacobian) @ operator.T
        x_t = torch.randn(args.image_dim, generator=generator, dtype=dtype)

        def denoiser(value, _sigma):
            return jacobian @ value

        exact_greedy = _exact_greedy(
            exact, args.noise_variance, args.budget
        )
        exact_value = _information_value(
            exact, exact_greedy, args.noise_variance
        )
        wrong_row_energy = (exact @ exact).diag() / (sigma**2)
        wrong_error = _relative_error(wrong_row_energy, exact.diag())

        for probe_count in args.probes:
            sketch = sketch_operator_posterior_covariance(
                x_t=x_t,
                mu=denoiser(x_t, sigma),
                sigma_t=sigma,
                denoiser=denoiser,
                forward=lambda value: operator @ value,
                adjoint=lambda value: operator.T @ value,
                measurement_template=torch.zeros(
                    args.measurement_dim, dtype=dtype
                ),
                num_probes=probe_count,
                fd_step=args.fd_step,
                generator=torch.Generator(device="cpu").manual_seed(
                    10_000 + seed
                ),
            )
            approximate = (
                sketch.measurement_factor
                @ sketch.measurement_factor.T
                + torch.diag(sketch.diagonal_residual_real)
            )
            atom_information = build_atom_information(
                sketch,
                atom_ids=torch.arange(args.measurement_dim),
                noise_variance=args.noise_variance,
                complex_noise_is_total=False,
            )
            selected, _ = greedy_logdet_select(
                atom_information, args.budget
            )
            selected_value = _information_value(
                exact, selected, args.noise_variance
            )
            rows.append({
                "seed": seed,
                "probes": probe_count,
                "effective_rank": sketch.rank,
                "diagonal_relative_error": _relative_error(
                    sketch.diagonal_hutchinson_real, exact.diag()
                ),
                "covariance_relative_error": _relative_error(
                    approximate, exact
                ),
                "legacy_row_energy_relative_error": wrong_error,
                "selection_overlap": (
                    len(set(selected) & set(exact_greedy))
                    / max(1, args.budget)
                ),
                "selection_regret": exact_value - selected_value,
                "core_asymmetry": sketch.diagnostics["core_asymmetry"],
                "core_negative_mass": sketch.diagnostics[
                    "core_negative_mass"
                ],
            })

    report = {
        "status": "ANALYTIC_VALIDATION",
        "git_commit": _git_commit(),
        "config": vars(args),
        "rows": rows,
        "summary_by_probes": _summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report["summary_by_probes"], indent=2))
    print(f"saved {output}")


def _exact_greedy(
    covariance: torch.Tensor,
    noise_variance: float,
    budget: int,
) -> list[int]:
    selected: list[int] = []
    remaining = list(range(covariance.shape[0]))
    for _ in range(min(budget, len(remaining))):
        best = max(
            remaining,
            key=lambda atom: _information_value(
                covariance, selected + [atom], noise_variance
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _information_value(
    covariance: torch.Tensor,
    selected: list[int],
    noise_variance: float,
) -> float:
    if not selected:
        return 0.0
    index = torch.as_tensor(selected, dtype=torch.long)
    submatrix = covariance[index][:, index]
    eye = torch.eye(
        len(selected), dtype=covariance.dtype, device=covariance.device
    )
    sign, value = torch.linalg.slogdet(
        eye + submatrix / float(noise_variance)
    )
    if float(sign.item()) <= 0:
        return float("-inf")
    return 0.5 * float(value.item())


def _relative_error(estimate: torch.Tensor, target: torch.Tensor) -> float:
    return float(
        ((estimate - target).norm() / target.norm().clamp_min(1e-15)).item()
    )


def _summarize(rows: list[dict]) -> dict:
    output = {}
    for probes in sorted({row["probes"] for row in rows}):
        subset = [row for row in rows if row["probes"] == probes]
        output[str(probes)] = {
            key: float(sum(row[key] for row in subset) / len(subset))
            for key in [
                "effective_rank",
                "diagonal_relative_error",
                "covariance_relative_error",
                "legacy_row_energy_relative_error",
                "selection_overlap",
                "selection_regret",
                "core_asymmetry",
                "core_negative_mass",
            ]
        }
    return output


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
