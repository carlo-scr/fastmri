#!/usr/bin/env python3
"""Fail-closed validation for corrected SketchSense production artifacts.

This validator never edits raw result JSON and never promotes it to
``VALIDATED``.  It checks the predeclared two-domain core gate and emits a
separate review manifest.  Submission readiness remains false until the
paper-critical external baselines and manual scientific review are complete.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_sha256(value) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value or "")))


def _positive_interval(interval) -> bool:
    return (
        isinstance(interval, list)
        and len(interval) == 2
        and _finite(interval[0])
        and _finite(interval[1])
        and 0.0 < float(interval[0]) <= float(interval[1])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_result", type=Path, required=True)
    parser.add_argument(
        "--mri_results",
        type=Path,
        nargs="+",
        required=True,
        help="Corrected R4/R8 x L8/L16 production results.",
    )
    parser.add_argument("--expected_commit", required=True)
    parser.add_argument("--expected_guided_commit", required=True)
    parser.add_argument("--expected_adps_commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, object] = {}

    def require(condition: bool, message: str) -> None:
        checks[message] = bool(condition)
        if not condition:
            failures.append(message)

    imagenet = _load(args.imagenet_result)
    require(
        imagenet.get("git_commit") == args.expected_commit,
        "ImageNet result uses the expected Git commit",
    )
    require(
        imagenet.get("status") == "CORRECTED_RUN_UNREVIEWED",
        "ImageNet raw result retains unreviewed status",
    )
    require(
        imagenet.get("protocol", {}).get("production_evidence_ready") is True,
        "ImageNet production protocol is complete",
    )
    require(
        imagenet.get("source_provenance", {}).get(
            "tracked_worktree_clean"
        ) is True,
        "ImageNet runner used a clean tracked worktree",
    )
    image_config = imagenet.get("config", {})
    require(
        int(image_config.get("n_images", 0)) >= 250,
        "ImageNet includes at least 250 held-out images",
    )
    require(
        int(image_config.get("sample_offset", 0)) >= 16,
        "ImageNet headline IDs exclude the reserved calibration prefix",
    )
    image_manifest = imagenet.get("sample_manifest", [])
    image_hashes = [row.get("sha256") for row in image_manifest]
    image_paths = [row.get("relative_path") for row in image_manifest]
    require(
        len(image_manifest) == int(image_config.get("n_images", -1)),
        "ImageNet sample manifest count matches the configuration",
    )
    require(
        len(set(image_paths)) == len(image_paths)
        and all(image_paths)
        and all(_is_sha256(value) for value in image_hashes),
        "ImageNet sample paths are unique and hashes are complete",
    )
    require(
        str(
            imagenet.get("model_provenance", {}).get(
                "checkpoint_sha256", ""
            )
        ).lower()
        == str(image_config.get("expected_checkpoint_sha256", "")).lower(),
        "ImageNet checkpoint matches the declared digest",
    )
    require(
        len(
            str(
                (imagenet.get("lpips_provenance") or {}).get(
                    "state_dict_sha256", ""
                )
            )
        )
        == 64,
        "ImageNet LPIPS state is hashed",
    )
    require(
        imagenet.get("model_provenance", {}).get("source_commit")
        == args.expected_guided_commit,
        "ImageNet guided-diffusion source commit is pinned",
    )
    dataset_provenance = imagenet.get("dataset_provenance", {})
    require(
        dataset_provenance.get("canonical_basenames_verified") is True
        and int(dataset_provenance.get("expected_count", 0)) == 50_000,
        "ImageNet canonical validation layout is verified",
    )
    required_image_policies = {
        "static",
        "random",
        "operator_diag",
        "sketch_marginal",
        "sketch_logdet",
        "ensemble",
        "ens_random",
        "oracle",
    }
    per_image = imagenet.get("per_image", {})
    require(
        set(per_image) == required_image_policies,
        "ImageNet result contains the exact required internal policy set",
    )
    expected_image_count = int(image_config.get("n_images", 0))
    expected_block_budget = int(image_config.get("budget_blocks", -1))
    for policy in required_image_policies:
        row = per_image.get(policy, {})
        require(
            all(
                len(row.get(key, [])) == expected_image_count
                for key in ["psnr", "ssim", "system_psnr", "system_ssim"]
            ),
            f"ImageNet {policy} metric arrays cover every image",
        )
        atom_sets = row.get("final_atom_ids", [])
        require(
            len(atom_sets) == expected_image_count
            and all(
                len(set(atom_ids)) == expected_block_budget
                for atom_ids in atom_sets
            ),
            f"ImageNet {policy} uses the exact per-image atom budget",
        )
    image_delta = imagenet.get("summary", {}).get(
        "delta_sketch_logdet_vs_sketch_marginal",
        {},
    )
    image_ci = image_delta.get("paired_bootstrap_95ci", [None, None])
    require(
        _positive_interval(image_ci),
        "ImageNet joint-vs-marginal paired interval is positive",
    )
    image_effect = image_delta.get("mean")
    require(
        _finite(image_effect)
        and float(image_effect) > 0
        and int(image_delta.get("n", 0)) == expected_image_count,
        "ImageNet joint-vs-marginal mean effect is finite",
    )
    for policy in [
        "operator_diag",
        "sketch_marginal",
        "sketch_logdet",
    ]:
        policy_summary = imagenet.get("summary", {}).get(policy, {})
        require(
            int(policy_summary.get("sketch_diagnostic_count", 0))
            == expected_image_count
            * int(image_config.get("active_rounds", 0)),
            f"ImageNet {policy} serializes per-image sketch diagnostics",
        )
        diagnostics = per_image.get(policy, {}).get(
            "sketch_diagnostics", []
        )
        require(
            {
                int(row.get("global_image_index", -1))
                for row in diagnostics
            }
            == set(range(
                int(image_config.get("sample_offset", 0)),
                int(image_config.get("sample_offset", 0))
                + expected_image_count,
            ))
            and all(
                _finite(row.get("core_asymmetry"))
                and _finite(row.get("core_negative_mass"))
                and _finite(row.get("effective_rank"))
                for row in diagnostics
            ),
            f"ImageNet {policy} sketch diagnostics are finite and cover IDs",
        )
        require(
            _finite(policy_summary.get("system_nfe_per_image")),
            f"ImageNet {policy} records finite actual NFE",
        )

    mri_artifacts = []
    mri_by_setting: dict[tuple[int, int], dict] = {}
    cohort_signatures = set()
    checkpoint_signatures = set()
    adps_signatures = set()
    for path in args.mri_results:
        result = _load(path)
        config = result.get("config", {})
        setting = (
            int(config.get("acceleration", -1)),
            int(config.get("active_lines", -1)),
        )
        require(
            setting not in mri_by_setting,
            f"MRI setting {setting} appears once",
        )
        mri_by_setting[setting] = result
        require(
            result.get("git_commit") == args.expected_commit,
            f"MRI setting {setting} uses the expected Git commit",
        )
        require(
            result.get("status") == "CORRECTED_RUN_UNREVIEWED",
            f"MRI setting {setting} raw result retains unreviewed status",
        )
        protocol = result.get("protocol", {})
        require(
            protocol.get("production_evidence_ready") is True,
            f"MRI setting {setting} production protocol is complete",
        )
        require(
            result.get("source_provenance", {}).get(
                "tracked_worktree_clean"
            ) is True,
            f"MRI setting {setting} used a clean tracked worktree",
        )
        require(
            protocol.get("full_volume_metrics") is True,
            f"MRI setting {setting} evaluates every selected volume slice",
        )
        require(
            int(config.get("num_volumes", 0)) >= 20,
            f"MRI setting {setting} includes at least 20 independent volumes",
        )
        require(
            len(result.get("volume_manifest", []))
            == int(config.get("num_volumes", -1)),
            f"MRI setting {setting} volume manifest count is exact",
        )
        cohort_provenance = result.get("volume_manifest_provenance") or {}
        actual_volumes = result.get("volume_manifest", [])
        volume_names = [row.get("volume") for row in actual_volumes]
        volume_hashes = [row.get("sha256") for row in actual_volumes]
        require(
            cohort_provenance.get("anatomy") == "brain"
            and cohort_provenance.get("split") == "validation",
            f"MRI setting {setting} declares brain validation data",
        )
        require(
            len(set(volume_names)) == len(volume_names)
            and all(volume_names)
            and len(set(volume_hashes)) == len(volume_hashes)
            and all(_is_sha256(value) for value in volume_hashes),
            f"MRI setting {setting} volume IDs and hashes are unique",
        )
        cohort_signatures.add((
            cohort_provenance.get("sha256"),
            tuple(row.get("volume") for row in actual_volumes),
            tuple(row.get("sha256") for row in actual_volumes),
        ))
        checkpoint_provenance = (
            result.get("checkpoint_manifest_provenance") or {}
        )
        checkpoint_signatures.add((
            checkpoint_provenance.get("sha256"),
            tuple(
                (row.get("relative_path"), row.get("sha256"))
                for row in result.get("model_manifest", [])
            ),
        ))
        adps_provenance = result.get("adps_provenance") or {}
        adps_signatures.add((
            adps_provenance.get("commit"),
            adps_provenance.get("tracked_worktree_clean"),
        ))
        require(
            adps_provenance.get("commit") == args.expected_adps_commit
            and adps_provenance.get("tracked_worktree_clean") is True,
            f"MRI setting {setting} uses the pinned clean ADPS source",
        )
        active_policies = [
            "static_match",
            "random_adaptive",
            "operator_diag_active",
            "sketch_marginal_active",
            "sketch_logdet_active",
            "ens_active",
            "ens_random_active",
            "oracle_active",
        ]
        summary = result.get("summary", {})
        per_slice = result.get("per_slice", [])
        expected_slice_count = len(per_slice)
        require(
            expected_slice_count > 0
            and len({row.get("volume") for row in per_slice})
            == int(config.get("num_volumes", 0)),
            f"MRI setting {setting} covers every declared volume",
        )
        width = int(config.get("target_resolution", [0, 0])[-1])
        active_lines = int(config.get("active_lines", -1))
        for row_index, slice_row in enumerate(per_slice):
            methods = slice_row.get("methods", {})
            expected_budget = min(
                width,
                int(slice_row.get("n_R_cols", -1)) + active_lines,
            )
            require(
                all(policy in methods for policy in active_policies),
                f"MRI setting {setting} slice {row_index} has all active policies",
            )
            budgets = [
                methods.get(policy, {}).get("budget_cols")
                for policy in active_policies
            ]
            final_counts = [
                len(methods.get(policy, {}).get("final_columns", []))
                for policy in active_policies
            ]
            require(
                all(value == expected_budget for value in budgets)
                and all(value == expected_budget for value in final_counts),
                f"MRI setting {setting} slice {row_index} has exact matched budgets",
            )
        for policy in active_policies:
            require(
                int(summary.get(policy, {}).get("n_volumes", 0))
                == int(config.get("num_volumes", -1))
                and int(summary.get(policy, {}).get("n", 0))
                == expected_slice_count,
                f"MRI setting {setting} {policy} covers all slices and volumes",
            )
        for policy in [
            "operator_diag_active",
            "sketch_marginal_active",
            "sketch_logdet_active",
        ]:
            diagnostic_rows = [
                row["methods"][policy]
                for row in result.get("per_slice", [])
                if policy in row.get("methods", {})
            ]
            require(
                len(diagnostic_rows) == expected_slice_count
                and all(
                    "sketch_core_asymmetry_mean" in row
                    and "sketch_negative_mass_mean" in row
                    and _finite(row["sketch_core_asymmetry_mean"])
                    and _finite(row["sketch_negative_mass_mean"])
                    and _finite(row.get("sketch_effective_rank_mean"))
                    for row in diagnostic_rows
                ),
                f"MRI setting {setting} {policy} serializes sketch diagnostics",
            )
        mri_artifacts.append({
            "path": str(path),
            "sha256": _sha256(path),
            "setting": list(setting),
        })

    expected_settings = {(4, 8), (4, 16), (8, 8), (8, 16)}
    require(
        set(mri_by_setting) == expected_settings,
        "MRI grid contains exactly R4/R8 by L8/L16",
    )
    require(
        len(cohort_signatures) == 1
        and next(iter(cohort_signatures))[0] is not None,
        "All MRI settings use one identical ordered cohort manifest",
    )
    require(
        len(checkpoint_signatures) == 1
        and next(iter(checkpoint_signatures))[0] is not None,
        "All MRI settings use one identical checkpoint manifest",
    )
    require(
        adps_signatures == {(args.expected_adps_commit, True)},
        "All MRI settings use one identical clean ADPS checkout",
    )
    primary_mri = mri_by_setting.get((4, 8), {})
    mri_delta = primary_mri.get("summary", {}).get(
        "_sketch_logdet_active_vs_sketch_marginal_active",
        {},
    )
    mri_ci = mri_delta.get("paired_bootstrap_95ci", [None, None])
    require(
        _positive_interval(mri_ci),
        "Primary MRI R4/L8 joint-vs-marginal paired interval is positive",
    )
    mri_effect = mri_delta.get("delta_selection_psnr_rss_mean")
    require(
        _finite(mri_effect)
        and float(mri_effect) > 0
        and int(mri_delta.get("n_volumes", 0)) == 20,
        "Primary MRI R4/L8 joint-vs-marginal mean effect is finite",
    )
    require(
        _finite(image_effect)
        and _finite(mri_effect)
        and max(float(image_effect), float(mri_effect)) >= 0.2,
        "At least one headline domain gains 0.2 dB",
    )

    blockers = [
        "Faithful ADS and AdaSense comparisons are not implemented.",
        "Faithful diffusion MRI BED (Iollo et al.) is not implemented.",
        "Free Hunch and Christoffel-DPS common-operator comparisons are pending.",
        "Ensemble K=8/16 and final compute-frontier runs are pending.",
        "Independent manual review of reconstructions, manifests, licenses, and "
        "paper claims is pending.",
    ]
    core_gate_passed = not failures
    report = {
        "status": (
            "CORE_GATE_PASS_SUBMISSION_BLOCKED"
            if core_gate_passed
            else "CORE_GATE_FAIL"
        ),
        "submission_ready": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "core_gate_passed": core_gate_passed,
        "checks": checks,
        "failures": failures,
        "submission_blockers": blockers,
        "artifacts": {
            "imagenet": {
                "path": str(args.imagenet_result),
                "sha256": _sha256(args.imagenet_result),
            },
            "mri": mri_artifacts,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not core_gate_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
