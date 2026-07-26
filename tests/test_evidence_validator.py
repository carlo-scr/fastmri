import json
import subprocess
import sys
from pathlib import Path


POLICIES = [
    "static_match",
    "random_adaptive",
    "operator_diag_active",
    "sketch_marginal_active",
    "sketch_logdet_active",
    "ens_active",
    "ens_random_active",
    "oracle_active",
]
IMAGE_POLICIES = [
    "static",
    "random",
    "operator_diag",
    "sketch_marginal",
    "sketch_logdet",
    "ensemble",
    "ens_random",
    "oracle",
]
GUIDED_COMMIT = "1" * 40
ADPS_COMMIT = "2" * 40


def _imagenet_result(commit):
    checkpoint_hash = "a" * 64
    summary = {
        "delta_sketch_logdet_vs_sketch_marginal": {
            "mean": 0.21,
            "paired_bootstrap_95ci": [0.01, 0.4],
            "n": 250,
        },
    }
    for policy in ["operator_diag", "sketch_marginal", "sketch_logdet"]:
        summary[policy] = {
            "sketch_diagnostic_count": 250,
            "system_nfe_per_image": 58.0,
        }
    per_image = {}
    for policy in IMAGE_POLICIES:
        row = {
            "psnr": [30.0] * 250,
            "ssim": [0.9] * 250,
            "system_psnr": [30.0] * 250,
            "system_ssim": [0.9] * 250,
            "final_atom_ids": [list(range(32)) for _ in range(250)],
            "sketch_diagnostics": [],
        }
        if policy in {"operator_diag", "sketch_marginal", "sketch_logdet"}:
            row["sketch_diagnostics"] = [
                {
                    "global_image_index": index,
                    "core_asymmetry": 0.0,
                    "core_negative_mass": 0.0,
                    "effective_rank": 4,
                }
                for index in range(16, 266)
            ]
        per_image[policy] = row
    return {
        "status": "CORRECTED_RUN_UNREVIEWED",
        "git_commit": commit,
        "protocol": {"production_evidence_ready": True},
        "source_provenance": {"tracked_worktree_clean": True},
        "config": {
            "n_images": 250,
            "sample_offset": 16,
            "budget_blocks": 32,
            "active_rounds": 1,
            "expected_checkpoint_sha256": checkpoint_hash,
        },
        "sample_manifest": [
            {
                "relative_path": f"ILSVRC2012_val_{index + 1:08d}.JPEG",
                "sha256": f"{index:064x}",
            }
            for index in range(250)
        ],
        "dataset_provenance": {
            "canonical_basenames_verified": True,
            "expected_count": 50_000,
        },
        "model_provenance": {
            "checkpoint_sha256": checkpoint_hash,
            "source_commit": GUIDED_COMMIT,
        },
        "lpips_provenance": {"state_dict_sha256": "b" * 64},
        "per_image": per_image,
        "summary": summary,
    }


def _mri_result(commit, acceleration, active_lines, cohort_digest="c" * 64):
    expected_budget = 32 + active_lines
    volumes = [
        {
            "volume": f"brain_{index:02d}.h5",
            "sha256": f"{index + 1:064x}",
        }
        for index in range(20)
    ]
    summary = {
        policy: {
            "budget_cols": expected_budget,
            "n_volumes": 20,
            "n": 20,
        }
        for policy in POLICIES
    }
    summary[
        "_sketch_logdet_active_vs_sketch_marginal_active"
    ] = {
        "delta_selection_psnr_rss_mean": 0.1,
        "paired_bootstrap_95ci": [0.01, 0.2],
        "n_volumes": 20,
    }
    diagnostic = {
        "sketch_core_asymmetry_mean": 0.0,
        "sketch_negative_mass_mean": 0.0,
    }
    per_slice = []
    for volume in volumes:
        methods = {
            policy: {
                "budget_cols": expected_budget,
                "final_columns": list(range(expected_budget)),
            }
            for policy in POLICIES
        }
        for policy in [
            "operator_diag_active",
            "sketch_marginal_active",
            "sketch_logdet_active",
        ]:
            methods[policy].update(diagnostic)
            methods[policy]["sketch_effective_rank_mean"] = 4.0
        per_slice.append({
            "volume": volume["volume"],
            "n_R_cols": 32,
            "methods": methods,
        })
    return {
        "status": "CORRECTED_RUN_UNREVIEWED",
        "git_commit": commit,
        "source_provenance": {"tracked_worktree_clean": True},
        "protocol": {
            "production_evidence_ready": True,
            "full_volume_metrics": True,
        },
        "config": {
            "acceleration": acceleration,
            "active_lines": active_lines,
            "num_volumes": 20,
            "target_resolution": [384, 320],
        },
        "volume_manifest_provenance": {
            "sha256": cohort_digest,
            "anatomy": "brain",
            "split": "validation",
        },
        "checkpoint_manifest_provenance": {"sha256": "d" * 64},
        "adps_provenance": {
            "commit": ADPS_COMMIT,
            "tracked_worktree_clean": True,
        },
        "volume_manifest": volumes,
        "model_manifest": [
            {"relative_path": "network-snapshot.pkl", "sha256": "e" * 64},
        ],
        "per_slice": per_slice,
        "summary": summary,
    }


def _write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def _run_validator(tmp_path, *, mixed_cohort=False):
    commit = "f" * 40
    image_path = _write_json(
        tmp_path / "imagenet.json",
        _imagenet_result(commit),
    )
    mri_paths = []
    for index, setting in enumerate([(4, 8), (4, 16), (8, 8), (8, 16)]):
        cohort_digest = (
            "9" * 64 if mixed_cohort and index == 3 else "c" * 64
        )
        mri_paths.append(_write_json(
            tmp_path / f"mri_{setting[0]}_{setting[1]}.json",
            _mri_result(commit, *setting, cohort_digest=cohort_digest),
        ))
    output = tmp_path / "validation.json"
    command = [
        sys.executable,
        "scripts/validate_sketchsense_evidence.py",
        "--imagenet_result",
        str(image_path),
        "--mri_results",
        *(str(path) for path in mri_paths),
        "--expected_commit",
        commit,
        "--expected_guided_commit",
        GUIDED_COMMIT,
        "--expected_adps_commit",
        ADPS_COMMIT,
        "--output",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    return result, output


def test_evidence_validator_passes_core_but_blocks_submission(tmp_path):
    result, output = _run_validator(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "CORE_GATE_PASS_SUBMISSION_BLOCKED"
    assert report["core_gate_passed"] is True
    assert report["submission_ready"] is False
    assert report["submission_blockers"]


def test_evidence_validator_rejects_mixed_mri_cohorts(tmp_path):
    result, output = _run_validator(tmp_path, mixed_cohort=True)
    assert result.returncode == 1
    report = json.loads(output.read_text())
    assert report["core_gate_passed"] is False
    assert (
        "All MRI settings use one identical ordered cohort manifest"
        in report["failures"]
    )
