"""Build the reproducible WACV 2027 SketchSense Colab handoff.

The generated notebook is intentionally unexecuted locally: its substantive
cells require licensed datasets, private checkpoints, Google Drive, and a
CUDA GPU. Run this builder after changing notebook prose or commands:

    python scripts/build_sketchsense_colab.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "colab_sketchsense_wacv27.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str, *, tags: list[str] | None = None):
    cell = nbf.v4.new_code_cell(source.strip())
    if tags:
        cell.metadata["tags"] = tags
    return cell


cells = [
    markdown(
        r"""
# SketchSense — WACV 2027 reproducible experiment handoff

**Mode:** experiment log/tutorial. **Status:** `PENDING_GPU_EXECUTION`.

This notebook evaluates *Operator-Space Diffusion Covariance Sketches for
Correlation-Aware Active Sensing*. It runs the corrected
analytic checks, a CIFAR development gate, multi-coil fastMRI pilots and
production runs, and a held-out ImageNet-256 experiment with OpenAI's official
unconditional ADM checkpoint.

No legacy PVAS result is imported. A result is not paper-valid merely because
the command finishes: inspect provenance, final budgets, NFE, PSD diagnostics,
paired effects, and the predeclared go/no-go gates.

The notebook does **not** download ImageNet or fastMRI. Point it at copies you
are licensed to use. Every headline run keeps system reconstruction separate
from a canonical passive replay of the selected final mask. Before running,
freeze 20 explicit brain-validation filenames with
`scripts/make_fastmri_cohort_manifest.py`, and fill the three expected model
SHA256 parameters. Placeholder hashes intentionally stop execution.
"""
    ),
    markdown(
        """
## Goal

Decide whether the correlation-aware SketchSense claim survives two domains:

1. corrected cross-Hutchinson and PSD sketch mathematics;
2. a measurable advantage for joint log-determinant selection over both the
   corrected diagonal and low-rank marginal policies;
3. a useful NFE/memory frontier versus particle acquisition;
4. stable results at volume level in MRI and over at least 250 held-out
   256-pixel natural images.
"""
    ),
    markdown("## Setup"),
    code(
        """
# 1. GPU and runtime
import os, platform, subprocess, sys, torch
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "Select a T4/A100 GPU runtime before continuing."
print("GPU:", torch.cuda.get_device_name(0))
subprocess.run(["nvidia-smi"], check=False)
"""
    ),
    code(
        """
# 2. Parameters — edit these once
REPO_URL = "https://github.com/carlo-scr/fastmri.git"
BRANCH = "research/sketchsense-wacv27"
REPO_COMMIT = "d0c7822b2875a4cb35303f6542c914d235e58f58"
REPO_DIR = "/content/fastmri"
GUIDED_DIFFUSION_COMMIT = "22e0df8183507e13a7813f8d38d51b072ca1e67c"
ADPS_COMMIT = "75feb2fad4b53e48791fcb0257f2e53a25340774"

DRIVE_ROOT = "/content/drive/MyDrive/sketchsense_wacv27"
MRI_SOURCE_DIR = "/content/drive/MyDrive/fastmri_artifacts/multicoil_val"
MRI_CHECKPOINT_SOURCE = "/content/drive/MyDrive/fastmri_artifacts/network-snapshot.pkl"
MRI_TRAINING_OPTIONS_SOURCE = "/content/drive/MyDrive/fastmri_artifacts/training_options.json"
MRI_COHORT_MANIFEST_SOURCE = (
    "/content/drive/MyDrive/fastmri_artifacts/sketchsense_brain_val20.json"
)
MRI_NETWORK_SHA256 = "SET_EXPECTED_NETWORK_SNAPSHOT_SHA256"
MRI_TRAINING_OPTIONS_SHA256 = "SET_EXPECTED_TRAINING_OPTIONS_SHA256"
MRI_NUM_VOLUMES = 20
MRI_SLICES_PER_VOLUME = 0  # Full volumes for stacked-volume headline metrics.

# Must be an explicit held-out ImageNet validation-image directory.
IMAGENET_VAL_DIR = "/content/drive/MyDrive/imagenet/ILSVRC2012_img_val"
ADM_CHECKPOINT_SHA256 = "SET_EXPECTED_OPENAI_ADM_SHA256"

RUN_PRODUCTION = False  # Flip only after both pilot gates are reviewed.
GLOBAL_SEED = 0
"""
        ,
        tags=["parameters"],
    ),
    code(
        """
# 3. Mount Drive, check out the immutable experiment commit, and install
from google.colab import drive
drive.mount("/content/drive")

from pathlib import Path
Path(DRIVE_ROOT).mkdir(parents=True, exist_ok=True)
if not Path(REPO_DIR, ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout",
         REPO_URL, REPO_DIR],
        check=True,
    )
subprocess.run(
    ["git", "-C", REPO_DIR, "fetch", "origin", BRANCH],
    check=True,
)
subprocess.run(
    ["git", "-C", REPO_DIR, "checkout", "--detach", REPO_COMMIT],
    check=True,
)
assert not subprocess.check_output(
    ["git", "-C", REPO_DIR, "status", "--porcelain", "--untracked-files=no"],
    text=True,
).strip(), "Tracked checkout is dirty; restart from a clean Colab runtime."
os.chdir(REPO_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)

# Official ADM implementation and EDM pickle dependencies, pinned by commit.
external_repositories = [
    ("https://github.com/openai/guided-diffusion.git",
     "external/guided-diffusion", GUIDED_DIFFUSION_COMMIT),
    ("https://github.com/utcsilab/ambient-diffusion-mri.git",
     "external/adps", ADPS_COMMIT),
]
for url, destination, revision in external_repositories:
    if not Path(destination, ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             url, destination],
            check=True,
        )
    subprocess.run(
        ["git", "-C", destination, "fetch", "--depth", "1", "origin", revision],
        check=True,
    )
    subprocess.run(
        ["git", "-C", destination, "checkout", "--detach", revision],
        check=True,
    )
"""
    ),
    code(
        """
# 4. Freeze code/environment provenance before any experiment
import json, hashlib, datetime
COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()
assert COMMIT == REPO_COMMIT, (COMMIT, REPO_COMMIT)
provenance = {
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repo_url": REPO_URL,
    "source_branch": BRANCH,
    "pinned_git_commit": REPO_COMMIT,
    "git_commit": COMMIT,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "guided_diffusion_commit": GUIDED_DIFFUSION_COMMIT,
    "adps_commit": ADPS_COMMIT,
}
Path("outputs/sketchsense").mkdir(parents=True, exist_ok=True)
provenance_path = Path("outputs/sketchsense/colab_provenance.json")
assert not provenance_path.exists(), (
    "This runtime already contains an experiment; restart to avoid mixing runs."
)
provenance_path.write_text(
    json.dumps(provenance, indent=2)
)
with Path("outputs/sketchsense/environment_freeze.txt").open("w") as handle:
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=handle,
        check=True,
    )
print(json.dumps(provenance, indent=2))
"""
    ),
    markdown(
        """
### Key assumptions

- Central finite differences approximate the local denoiser Jacobian.
- The PSD projection is a local Gaussian approximation, not a claim that the
  learned denoiser Jacobian is exactly a covariance.
- MRI complex noise arguments denote total power, `E|n|²`; real and imaginary
  components each receive half.
- Licensed datasets are staged by the researcher. No training split or silent
  dataset fallback is allowed.
"""
    ),
    markdown("## Steps"),
    markdown("### A. Analytic and regression checks"),
    code(
        """
# 5. Exact real/complex covariance, PSD, grouping, budget, and seed regressions
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
     "tests/test_operator_sketch.py",
     "tests/test_posterior_var.py",
     "tests/test_protocol_regressions.py",
     "tests/test_evidence_validator.py"],
    check=True,
)
"""
    ),
    code(
        """
# 6. Measured convergence and selection regret on non-idempotent covariances
subprocess.run(
    [sys.executable, "scripts/validate_sketchsense.py",
     "--image_dim", "128", "--measurement_dim", "96",
     "--latent_rank", "16", "--probes", "1", "2", "4", "8", "16",
     "--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
     "--output", "outputs/sketchsense/analytic_validation.json"],
    check=True,
)
analytic = json.loads(
    Path("outputs/sketchsense/analytic_validation.json").read_text()
)
display(analytic["summary_by_probes"])
"""
    ),
    markdown("### B. CIFAR-10 development gate"),
    code(
        """
# 7. Cache only the documented CIFAR-10 test split and hash the artifact
from datasets import load_dataset
import numpy as np

cifar_cache = Path("data/cifar10/test_images_512.pt")
cifar_cache.parent.mkdir(parents=True, exist_ok=True)
if not cifar_cache.is_file():
    cifar = load_dataset("uoft-cs/cifar10", split="test")
    arrays = np.stack([np.asarray(image.convert("RGB")) for image in cifar[:512]["img"]])
    tensor = torch.from_numpy(arrays).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    torch.save(tensor, cifar_cache)
print("CIFAR cache SHA256:", hashlib.sha256(cifar_cache.read_bytes()).hexdigest())
"""
    ),
    code(
        """
# 8. Small pilot: canonical replay is ON; NFE and final masks are recorded
cifar_pilot = [
    sys.executable, "-m", "poc_natural.run_cifar",
    "--mode", "sr", "--n_images", "32", "--batch", "8",
    "--num_steps", "100", "--frac0", "0", "--block", "4",
    "--budget_blocks", "12", "--sigma_y", "0", "--active_rounds", "1",
    "--sketch_rank", "4", "--lpips",
    "--policies", "static", "random", "operator_diag",
    "sketch_marginal", "sketch_logdet", "ensemble", "ens_random", "oracle",
    "--tag", "sketchsense_pilot", "--device", "cuda", "--seed", str(GLOBAL_SEED),
]
subprocess.run(cifar_pilot, check=True)
"""
    ),
    code(
        """
# 9. Inspect the development gate; do not promote a failing pilot
import pandas as pd
cifar_result = json.loads(Path(
    "outputs/poc_natural/cifar_sketchsense_pilot/results.json"
).read_text())
cifar_rows = pd.DataFrame(cifar_result["summary"]).T
display(cifar_rows[[
    "psnr_mean", "system_psnr_mean", "system_nfe_per_image",
    "system_wall_time_seconds", "system_peak_cuda_memory_mb", "lpips_mean"
]].dropna(how="all"))
delta_logdet_marginal = (
    cifar_rows.loc["sketch_logdet", "psnr_mean"]
    - cifar_rows.loc["sketch_marginal", "psnr_mean"]
)
delta_logdet_diagonal = (
    cifar_rows.loc["sketch_logdet", "psnr_mean"]
    - cifar_rows.loc["operator_diag", "psnr_mean"]
)
print("Canonical Δ logdet-marginal:", delta_logdet_marginal, "dB")
print("Canonical Δ logdet-diagonal:", delta_logdet_diagonal, "dB")
"""
    ),
    code(
        """
# 10. Development-scale production arm (still not a headline domain)
if RUN_PRODUCTION:
    command = cifar_pilot.copy()
    command[command.index("32")] = "128"
    command[command.index("8")] = "16"
    command[command.index("sketchsense_pilot")] = "sketchsense_main"
    subprocess.run(command, check=True)
else:
    print("Skipped: set RUN_PRODUCTION=True only after reviewing the pilot.")
"""
    ),
    markdown("### C. Multi-coil fastMRI pilot and volume-level production"),
    code(
        """
# 11. Verify and stage the immutable MRI model/cohort manifests
import glob, shutil
def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

mri_checkpoint_dir = Path("checkpoints/edm/supervised_R=1")
mri_checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_inputs = [
    (MRI_CHECKPOINT_SOURCE, MRI_NETWORK_SHA256),
    (MRI_TRAINING_OPTIONS_SOURCE, MRI_TRAINING_OPTIONS_SHA256),
]
for source, _ in checkpoint_inputs:
    source_path = Path(source)
    assert source_path.is_file(), f"Missing required checkpoint artifact: {source}"
    print(source_path.name, "SHA256:", sha256_file(source_path))
assert not MRI_NETWORK_SHA256.startswith("SET_")
assert not MRI_TRAINING_OPTIONS_SHA256.startswith("SET_")
for source, expected_sha256 in checkpoint_inputs:
    source_path = Path(source)
    assert sha256_file(source_path) == expected_sha256.lower(), (
        f"Source checkpoint hash mismatch: {source}"
    )
    target = mri_checkpoint_dir / source_path.name
    if not target.exists():
        shutil.copy2(source_path, target)
    assert sha256_file(target) == expected_sha256.lower(), (
        f"Stale staged checkpoint: {target}; use a clean runtime."
    )

manifest_dir = Path("outputs/sketchsense/manifests")
manifest_dir.mkdir(parents=True, exist_ok=True)
mri_checkpoint_manifest = manifest_dir / "mri_checkpoint_manifest.json"
mri_checkpoint_manifest.write_text(json.dumps({
    "artifacts": [
        {"relative_path": Path(source).name, "sha256": expected}
        for source, expected in checkpoint_inputs
    ],
}, indent=2))

cohort_source = Path(MRI_COHORT_MANIFEST_SOURCE)
assert cohort_source.is_file(), "Create the frozen 20-volume cohort manifest first."
cohort_spec = json.loads(cohort_source.read_text())
assert cohort_spec.get("anatomy") == "brain"
assert cohort_spec.get("split") == "validation"
assert cohort_spec.get("dataset_revision") not in {None, "", "unspecified"}
cohort_rows = cohort_spec.get("volumes", [])
assert len(cohort_rows) == MRI_NUM_VOLUMES
cohort_names = [row["filename"] for row in cohort_rows]
cohort_hashes = [row["sha256"].lower() for row in cohort_rows]
assert len(set(cohort_names)) == MRI_NUM_VOLUMES
assert len(set(cohort_hashes)) == MRI_NUM_VOLUMES
assert all(Path(name).name == name for name in cohort_names)
MRI_DATASET_REVISION = cohort_spec["dataset_revision"]

mri_cohort_manifest = manifest_dir / "mri_cohort_manifest.json"
shutil.copy2(cohort_source, mri_cohort_manifest)
cohort_digest = sha256_file(mri_cohort_manifest)
mri_data_dir = Path("/content/sketchsense_inputs") / f"mri_{cohort_digest[:12]}"
mri_data_dir.mkdir(parents=True, exist_ok=True)
for row in cohort_rows:
    source = Path(MRI_SOURCE_DIR) / row["filename"]
    assert source.is_file(), f"Missing cohort volume: {source}"
    assert sha256_file(source) == row["sha256"].lower(), (
        f"Cohort hash mismatch: {source}"
    )
    target = mri_data_dir / source.name
    if not target.exists():
        target.symlink_to(source.resolve())
    assert target.resolve() == source.resolve()
assert {path.name for path in mri_data_dir.glob("*.h5")} == set(cohort_names)
print("Verified and staged", len(cohort_rows), "frozen brain volumes")
"""
    ),
    code(
        """
# 12. MRI pilot: two volumes x four center slices; hashes skipped only for speed
mri_pilot_dir = "outputs/sketchsense/mri_pilot_R4_L8"
mri_pilot = [
    sys.executable, "scripts/reconstruct_mc.py",
    "--checkpoint_dir", str(mri_checkpoint_dir),
    "--adps_root", "external/adps",
    "--expected_adps_commit", ADPS_COMMIT,
    "--data_path", str(mri_data_dir),
    "--dataset_revision", MRI_DATASET_REVISION,
    "--num_volumes", "2", "--num_slices", "4",
    "--acceleration", "4", "--center_fraction", "0.08",
    "--num_steps", "20", "--target_resolution", "384", "320",
    "--active_lines", "8", "--active_rounds", "1",
    "--sketch_rank", "4", "--device", "cuda",
    "--methods", "pigdm_mc", "static_match", "random_adaptive",
    "operator_diag_active", "sketch_marginal_active", "sketch_logdet_active",
    "ens_active", "oracle_active",
    "--skip_file_hashes", "--output_dir", mri_pilot_dir,
]
subprocess.run(mri_pilot, check=True)
"""
    ),
    code(
        """
# 13. MRI pilot check uses canonical selection metrics
mri_result = json.loads(Path(mri_pilot_dir, "results.json").read_text())
mri_rows = pd.DataFrame(mri_result["summary"]).T
display(mri_rows[[
    "selection_psnr_rss_volume_mean", "selection_ssim_rss_volume_mean",
    "selection_nmse_rss_volume_mean", "psnr_sense_mean", "nfe_mean",
    "runtime_s_mean", "peak_cuda_memory_mb_max", "budget_cols"
]].dropna(how="all"))
print(
    "Canonical Δ logdet-marginal:",
    mri_rows.loc["sketch_logdet_active", "selection_psnr_rss_volume_mean"]
    - mri_rows.loc["sketch_marginal_active", "selection_psnr_rss_volume_mean"],
    "dB",
)
"""
    ),
    code(
        """
# 14. Headline MRI grid: volume is the independent statistical unit
if RUN_PRODUCTION:
    assert MRI_SLICES_PER_VOLUME == 0, (
        "Headline fastMRI evidence must evaluate every slice in each volume."
    )
    for acceleration, active_lines in [(4, 8), (4, 16), (8, 8), (8, 16)]:
        output = f"outputs/sketchsense/mri_R{acceleration}_L{active_lines}"
        command = [
            sys.executable, "scripts/reconstruct_mc.py",
            "--checkpoint_dir", str(mri_checkpoint_dir),
            "--checkpoint_manifest", str(mri_checkpoint_manifest),
            "--adps_root", "external/adps",
            "--expected_adps_commit", ADPS_COMMIT,
            "--data_path", str(mri_data_dir),
            "--volume_manifest", str(mri_cohort_manifest),
            "--dataset_revision", MRI_DATASET_REVISION,
            "--anatomy", "brain",
            "--num_volumes", str(MRI_NUM_VOLUMES),
            "--num_slices", str(MRI_SLICES_PER_VOLUME),
            "--acceleration", str(acceleration),
            "--center_fraction", "0.08", "--num_steps", "20",
            "--target_resolution", "384", "320",
            "--active_lines", str(active_lines), "--active_rounds", "1",
            "--sketch_rank", "4", "--device", "cuda",
            "--methods", "pigdm_mc", "static_match", "random_adaptive",
            "operator_diag_active", "sketch_marginal_active",
            "sketch_logdet_active", "ens_active", "ens_random_active",
            "oracle_active", "--output_dir", output,
        ]
        subprocess.run(command, check=True)
        production_result = json.loads(
            Path(output, "results.json").read_text()
        )
        assert production_result["protocol"]["full_volume_metrics"]
        assert production_result["protocol"]["production_evidence_ready"]
        assert all(
            row.get("n_volumes") == MRI_NUM_VOLUMES
            for name, row in production_result["summary"].items()
            if not name.startswith("_")
        )
else:
    print("Skipped production MRI grid.")
"""
    ),
    markdown("### D. Held-out ImageNet-256 with official unconditional ADM"),
    code(
        """
# 15. Verify licensed validation data and download only OpenAI's public checkpoint
imagenet_root = Path(IMAGENET_VAL_DIR)
assert imagenet_root.is_dir(), "Point IMAGENET_VAL_DIR at the held-out validation images."
image_files = [
    path for path in imagenet_root.rglob("*")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
]
expected_names = {
    f"ILSVRC2012_val_{index:08d}.JPEG"
    for index in range(1, 50_001)
}
assert len(image_files) == 50_000
assert {path.name for path in image_files} == expected_names, (
    "ImageNet path must contain the canonical 50,000 validation basenames."
)

adm_checkpoint = Path("/content/models/256x256_diffusion_uncond.pt")
adm_checkpoint.parent.mkdir(parents=True, exist_ok=True)
if not adm_checkpoint.is_file():
    import urllib.request
    urllib.request.urlretrieve(
        "https://openaipublic.blob.core.windows.net/"
        "diffusion/jul-2021/256x256_diffusion_uncond.pt",
        adm_checkpoint,
    )
print("ADM SHA256:", hashlib.sha256(adm_checkpoint.read_bytes()).hexdigest())
assert not ADM_CHECKPOINT_SHA256.startswith("SET_")
assert sha256_file(adm_checkpoint) == ADM_CHECKPOINT_SHA256.lower()
"""
    ),
    code(
        """
# 16. ImageNet smoke/pilot; loader records every selected file hash
imagenet_pilot_dir = "outputs/sketchsense/imagenet256_pilot"
subprocess.run([
    sys.executable, "-m", "poc_natural.run_imagenet",
    "--image_dir", str(imagenet_root),
    "--checkpoint", str(adm_checkpoint),
    "--expected_checkpoint_sha256", ADM_CHECKPOINT_SHA256,
    "--guided_repo", "external/guided-diffusion",
    "--n_images", "16", "--sample_offset", "0",
    "--batch", "1", "--num_steps", "50",
    "--block", "16", "--budget_blocks", "32", "--active_rounds", "1",
    "--sketch_rank", "4", "--ensemble_k", "4", "--fp16", "--lpips",
    "--policies", "static", "random", "operator_diag", "sketch_marginal",
    "sketch_logdet", "ensemble", "ens_random", "oracle",
    "--device", "cuda", "--seed", str(GLOBAL_SEED),
    "--output_dir", imagenet_pilot_dir,
], check=True)
"""
    ),
    code(
        """
# 17. Inspect held-out pilot effects and provenance before scaling
imagenet_result = json.loads(
    Path(imagenet_pilot_dir, "results.json").read_text()
)
assert len(imagenet_result["sample_manifest"]) == 16
imagenet_rows = pd.DataFrame(imagenet_result["summary"]).T
display(imagenet_rows[[
    "psnr_mean", "system_psnr_mean", "lpips_mean", "system_nfe_per_image",
    "system_wall_time_seconds", "system_peak_cuda_memory_mb"
]].dropna(how="all"))
"""
    ),
    code(
        """
# 18. Headline natural-image run: at least 250 immutable held-out images
if RUN_PRODUCTION:
    subprocess.run([
        sys.executable, "-m", "poc_natural.run_imagenet",
        "--image_dir", str(imagenet_root),
        "--checkpoint", str(adm_checkpoint),
        "--expected_checkpoint_sha256", ADM_CHECKPOINT_SHA256,
        "--guided_repo", "external/guided-diffusion",
        "--n_images", "250", "--sample_offset", "16",
        "--batch", "2", "--num_steps", "50",
        "--block", "16", "--budget_blocks", "32", "--active_rounds", "1",
        "--sketch_rank", "4", "--ensemble_k", "4", "--fp16", "--lpips",
        "--policies", "static", "random", "operator_diag", "sketch_marginal",
        "sketch_logdet", "ensemble", "ens_random", "oracle",
        "--device", "cuda", "--seed", str(GLOBAL_SEED),
        "--output_dir", "outputs/sketchsense/imagenet256_n250",
    ], check=True)
    headline_imagenet_result = json.loads(
        Path(
            "outputs/sketchsense/imagenet256_n250/results.json"
        ).read_text()
    )
    assert headline_imagenet_result["protocol"]["production_evidence_ready"]
    assert headline_imagenet_result["git_commit"] == COMMIT
    assert len(headline_imagenet_result["sample_manifest"]) == 250
    pilot_hashes = {
        row["sha256"] for row in imagenet_result["sample_manifest"]
    }
    headline_hashes = {
        row["sha256"]
        for row in headline_imagenet_result["sample_manifest"]
    }
    assert pilot_hashes.isdisjoint(headline_hashes)
else:
    print("Skipped production ImageNet run.")
"""
    ),
    markdown("## Checks"),
    code(
        """
# 19. Cross-domain go/no-go report (pilot evidence only)
gate_report = {
    "git_commit": COMMIT,
    "cifar_logdet_minus_marginal_db": float(delta_logdet_marginal),
    "cifar_logdet_minus_diagonal_db": float(delta_logdet_diagonal),
    "mri_logdet_minus_marginal_db": float(
        mri_rows.loc["sketch_logdet_active", "selection_psnr_rss_volume_mean"]
        - mri_rows.loc["sketch_marginal_active", "selection_psnr_rss_volume_mean"]
    ),
    "paper_thresholds": {
        "headline_domain_gain_db": 0.2,
        "second_domain_nonnegative": True,
        "headline_natural_images": 250,
        "headline_mri_independent_volumes": MRI_NUM_VOLUMES,
    },
    "decision": "REVIEW_REQUIRED",
}
Path("outputs/sketchsense/pilot_gate_report.json").write_text(
    json.dumps(gate_report, indent=2)
)
print(json.dumps(gate_report, indent=2))

if RUN_PRODUCTION:
    production_validation = (
        f"outputs/sketchsense/evidence_validation_{COMMIT[:12]}.json"
    )
    subprocess.run([
        sys.executable, "scripts/validate_sketchsense_evidence.py",
        "--imagenet_result",
        "outputs/sketchsense/imagenet256_n250/results.json",
        "--mri_results",
        "outputs/sketchsense/mri_R4_L8/results.json",
        "outputs/sketchsense/mri_R4_L16/results.json",
        "outputs/sketchsense/mri_R8_L8/results.json",
        "outputs/sketchsense/mri_R8_L16/results.json",
        "--expected_commit", COMMIT,
        "--expected_guided_commit", GUIDED_DIFFUSION_COMMIT,
        "--expected_adps_commit", ADPS_COMMIT,
        "--output", production_validation,
    ], check=True)
    reviewed_core_gate = json.loads(Path(production_validation).read_text())
    assert reviewed_core_gate["submission_ready"] is False
    print(
        "Core gate:", reviewed_core_gate["status"],
        "— external-baseline blockers remain."
    )
"""
    ),
    code(
        """
# 20. Package configs/results/provenance to Drive with an archive hash
import tarfile
bundle = Path("/content/sketchsense_results.tar.gz")
assert not bundle.exists(), "Use a clean runtime; refusing to overwrite an archive."
with tarfile.open(bundle, "w:gz") as archive:
    for relative in [
        "outputs/sketchsense",
        "outputs/poc_natural/cifar_sketchsense_pilot",
        "docs/sketchsense_proposal.md",
        "docs/protocol_audit.md",
    ]:
        path = Path(relative)
        if path.exists():
            archive.add(path, arcname=relative)
bundle_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
destination = Path(DRIVE_ROOT) / f"sketchsense_{COMMIT[:12]}.tar.gz"
assert not destination.exists(), (
    f"Refusing to overwrite existing evidence bundle: {destination}"
)
shutil.copy2(bundle, destination)
hash_destination = Path(str(destination) + ".sha256")
assert not hash_destination.exists()
hash_destination.write_text(bundle_hash + "\\n")
print("Saved:", destination)
print("SHA256:", bundle_hash)
"""
    ),
    markdown(
        """
## Next Steps

1. Review the pilot JSON rather than the notebook's console summaries.
2. If the analytic gate fails, fix the implementation; do not tune around it.
3. If log-det is not positive in both domains, test rank/batch-size calibration
   before claiming correlations help.
4. Integrate and run Free Hunch, ADS/AdaSense-style, and Christoffel-DPS
   baselines in the same canonical-replay protocol.
5. Flip `RUN_PRODUCTION=True` only after the branch commit, dataset revisions,
   and pilot decision are frozen.
6. Promote `CORRECTED_RUN_UNREVIEWED` to `VALIDATED` only after checking input
   hashes, budgets, actual NFE, per-volume/per-image effects, and PSD defects.
"""
    ),
]

for cell_index, cell in enumerate(cells):
    cell["id"] = hashlib.sha256(
        f"{cell_index}\0{cell['cell_type']}\0{cell['source']}".encode("utf-8")
    ).hexdigest()[:12]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
