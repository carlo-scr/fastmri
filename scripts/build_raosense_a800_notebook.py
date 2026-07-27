"""Build the two-domain RaoSense A800 experiment notebook.

The generated notebook is intentionally unexecuted.  Real cells require
licensed fastMRI/KITTI-360 data, public model checkpoints, and CUDA hardware.

Run after editing this source:

    python scripts/build_raosense_a800_notebook.py
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "raosense_a800_wacv27.ipynb"


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
# RaoSense — WACV 2027 A800 experiment book

**Status:** `PENDING_GPU_AND_LICENSED_DATA`  
**Evidence policy:** synthetic checks may be reported as synthetic; no MRI or
LiDAR number is a paper result until the production gates at the end pass.

RaoSense represents a measurement-conditioned diffusion posterior by a small
mixture of local low-rank tangent Gaussians.  It evaluates likelihoods,
information bounds, Bayesian updates, and horizon-two fantasy outcomes in an
operator-sufficient subspace whose dimension is at most
\(\sum_k r_k + K - 1\), independent of the number of scalar measurements in a
candidate action.

This notebook evaluates that claim on two non-toy domains with deliberately
different sensing geometry:

1. raw multi-coil **fastMRI+ brain and knee**, using global complex
   Fourier/SENSE lines and pathology annotations;
2. **KITTI-360 programmable LiDAR** with the public R2DM range-image diffusion
   prior, using local spherical beam tiles.

It does not use CIFAR or ImageNet.  It can run on one CUDA GPU; when two GPUs
are visible, the brain and knee arms are launched concurrently.
"""
    ),
    markdown(
        r"""
## What would falsify the paper

These gates are fixed before inspecting production results.

- The mixture must improve predictive calibration or retrospective
  action-oracle Spearman by at least **0.10** over both a point-particle belief
  and the single-Gaussian SketchSense limit.
- \(H=2\) must improve over \(H=1\) with a positive paired,
  volume/drive-clustered 95% interval in **both** headline domains.
- \(H=2\) must route differently from \(H=1\) on at least **20%** of examples;
  otherwise the claimed outcome adaptivity is not operational.
- \(K=8,r=2\) must beat a matched-compute \(K=12,r=0\) particle control.
- The between-mode information fraction must be non-negligible at useful
  checkpoints.
- MRI promotion requires at least **0.3 dB** volume PSNR or a positive
  pathology-region/task gain over the strongest active-diffusion baseline.
- LiDAR promotion requires roughly **10%** relative range-MAE reduction or
  **2 semantic-mIoU points** at a matched angular-query budget.
- Canonical replay metrics, online-system metrics, NFE, wall time, and peak
  memory are always reported separately.

Failure means simplify or reposition the contribution; it does not mean tune
the threshold after seeing the test set.
"""
    ),
    markdown("## 1. Immutable parameters"),
    code(
        """
from pathlib import Path

# Source pins.  REPO_COMMIT contains the tested RaoSense MRI/LiDAR core.
REPO_URL = "https://github.com/carlo-scr/fastmri.git"
REPO_BRANCH = "research/sketchsense-wacv27"
REPO_COMMIT = "6f6bee8a319a971a6091628041b812f23466292a"
ADPS_COMMIT = "75feb2fad4b53e48791fcb0257f2e53a25340774"
R2DM_COMMIT = "ad66e71e86051d0341e096c60ece30a7de6c5b3f"
FASTMRI_PLUS_COMMIT = "67ed9a68ee33d3197f93893e8125688470c4a577"

R2DM_CHECKPOINT_URL = (
    "https://github.com/kazuto1011/r2dm/releases/download/weights/"
    "r2dm-h-kitti360-300k.pth"
)
R2DM_CHECKPOINT_SHA256 = (
    "89be3972a95a1e3ae9ae2cd8783b92162808e0acf577ff92d8c86a9b27db5396"
)

# "smoke": 2 MRI volumes/anatomy + 8 LiDAR frames
# "pilot": frozen 100 brain + 100 knee volumes + 1,000 LiDAR frames
# "production": full frozen annotated cohorts + 5,000 thinned LiDAR frames
RUN_STAGE = "smoke"
assert RUN_STAGE in {"smoke", "pilot", "production"}
RUN_EXPENSIVE = False  # Must be True before pilot/production cells execute.
GLOBAL_SEED = 2027

USE_GOOGLE_DRIVE = True
WORK_ROOT = Path("/content")
REPO_DIR = WORK_ROOT / "fastmri"
ARTIFACT_ROOT = WORK_ROOT / "raosense_artifacts"
DRIVE_EXPORT = Path(
    "/content/drive/MyDrive/raosense_wacv27"
)

# Licensed fastMRI inputs and fail-closed cohort manifests.
MRI_BRAIN_DATA = Path(
    "/content/drive/MyDrive/fastmri/multicoil_brain_val"
)
MRI_KNEE_DATA = Path(
    "/content/drive/MyDrive/fastmri/multicoil_knee_val"
)
MRI_BRAIN_MANIFEST = Path(
    "/content/drive/MyDrive/fastmri/manifests/brain_ANNOTATED_FROZEN.json"
)
MRI_KNEE_MANIFEST = Path(
    "/content/drive/MyDrive/fastmri/manifests/knee_ANNOTATED_FROZEN.json"
)
MRI_BRAIN_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/fastmri/checkpoints/brain_edm"
)
MRI_KNEE_CHECKPOINT_DIR = Path(
    "/content/drive/MyDrive/fastmri/checkpoints/knee_edm"
)
MRI_BRAIN_CHECKPOINT_MANIFEST = Path(
    "/content/drive/MyDrive/fastmri/manifests/brain_checkpoint.json"
)
MRI_KNEE_CHECKPOINT_MANIFEST = Path(
    "/content/drive/MyDrive/fastmri/manifests/knee_checkpoint.json"
)

# KITTI-360 registration/license is required.  This must point at
# dataset/data_3d_raw, not at a silently downloaded substitute.
KITTI360_DATA_ROOT = Path(
    "/content/drive/MyDrive/KITTI-360/dataset/data_3d_raw"
)
R2DM_CHECKPOINT = WORK_ROOT / "models/r2dm-h-kitti360-300k.pth"

# fastMRI+ annotations are cloned at a pinned commit below.
TARGET_MRI_VOLUMES = {
    "smoke": 2,
    "pilot": 100,
    "production": None,  # use every volume in each frozen manifest
}
TARGET_LIDAR_FRAMES = {"smoke": 8, "pilot": 1_000, "production": 5_000}
"""
        ,
        tags=["parameters"],
    ),
    markdown("## 2. Runtime, checkout, and provenance"),
    code(
        """
import datetime
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys

import numpy as np
import torch

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

assert torch.cuda.is_available(), "Attach at least one CUDA GPU."
GPU_COUNT = torch.cuda.device_count()
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch/CUDA:", torch.__version__, torch.version.cuda)
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)])
subprocess.run(["nvidia-smi"], check=False)

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
"""
    ),
    code(
        """
def checkout(url, destination, revision, branch=None):
    destination = Path(destination)
    if not (destination / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             url, str(destination)],
            check=True,
        )
    fetch_target = branch or revision
    subprocess.run(
        ["git", "-C", str(destination), "fetch", "origin", fetch_target],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == revision, (actual, revision)
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain",
         "--untracked-files=no"],
        text=True,
    ).strip()
    assert not dirty, f"Tracked checkout is dirty: {destination}"

checkout(REPO_URL, REPO_DIR, REPO_COMMIT, REPO_BRANCH)
os.chdir(REPO_DIR)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     "requirements.txt"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "datasets>=2.18", "pandas>=2.0", "pyarrow>=14", "scipy>=1.11",
     "einops>=0.7", "pydantic<3", "torchmetrics>=1.3"],
    check=True,
)

checkout(
    "https://github.com/utcsilab/ambient-diffusion-mri.git",
    REPO_DIR / "external/adps",
    ADPS_COMMIT,
)
checkout(
    "https://github.com/kazuto1011/r2dm.git",
    REPO_DIR / "external/r2dm",
    R2DM_COMMIT,
)
checkout(
    "https://github.com/microsoft/fastmri-plus.git",
    REPO_DIR / "external/fastmri-plus",
    FASTMRI_PLUS_COMMIT,
)
"""
    ),
    code(
        """
def sha256_file(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()

provenance = {
    "status": "PENDING_GPU_AND_LICENSED_DATA",
    "timestamp_utc": datetime.datetime.now(
        datetime.timezone.utc
    ).isoformat(),
    "run_stage": RUN_STAGE,
    "repo_commit": REPO_COMMIT,
    "adps_commit": ADPS_COMMIT,
    "r2dm_commit": R2DM_COMMIT,
    "fastmri_plus_commit": FASTMRI_PLUS_COMMIT,
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpus": [torch.cuda.get_device_name(i) for i in range(GPU_COUNT)],
}
provenance_path = ARTIFACT_ROOT / "provenance.json"
assert not provenance_path.exists(), (
    "Use a clean ARTIFACT_ROOT for each run; existing provenance found."
)
provenance_path.write_text(json.dumps(provenance, indent=2))
with (ARTIFACT_ROOT / "environment_freeze.txt").open("w") as handle:
    subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        stdout=handle,
        check=True,
    )
display(provenance)
"""
    ),
    markdown("## 3. CPU falsification suite before any GPU spend"),
    code(
        """
subprocess.run(
    [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
     "tests/test_rao_belief.py",
     "tests/test_joint_operator.py",
     "tests/test_raosense_mri.py",
     "tests/test_raosense_lidar.py",
     "tests/test_operator_sketch.py",
     "tests/test_posterior_var.py",
     "tests/test_protocol_regressions.py"],
    check=True,
)
subprocess.run(
    [sys.executable, "scripts/validate_raosense.py",
     "--output", str(ARTIFACT_ROOT / "analytic_validation.json")],
    check=True,
)
analytic = json.loads(
    (ARTIFACT_ROOT / "analytic_validation.json").read_text()
)
display(analytic)
"""
    ),
    markdown(
        r"""
The analytic file is the only evidence that exists before this notebook is
run.  It checks covariance-risk reduction, information lower/upper bounds,
sufficient-rank compression, a six-mode non-myopic trap, and the optimism of
pretending a nuisance operator is known.  It is not imaging evidence.
"""
    ),
    markdown("## 4. Fail-closed MRI data and checkpoint contracts"),
    code(
        """
def load_and_verify_volume_manifest(path, data_root, anatomy):
    path, data_root = Path(path), Path(data_root)
    assert path.is_file(), f"Missing frozen {anatomy} manifest: {path}"
    spec = json.loads(path.read_text())
    assert spec["dataset"] == "fastMRI"
    assert spec["split"] == "validation"
    assert spec["anatomy"] == anatomy
    assert spec["dataset_revision"] not in {"", "unspecified", None}
    rows = spec["volumes"]
    assert rows and len({row["filename"] for row in rows}) == len(rows)
    target_count = TARGET_MRI_VOLUMES[RUN_STAGE]
    if target_count is not None:
        assert len(rows) >= target_count, (len(rows), target_count)
        rows = rows[:target_count]
    for row in rows:
        source = data_root / row["filename"]
        assert source.is_file(), source
        assert sha256_file(source) == row["sha256"].lower(), source
    subset = dict(spec)
    subset["volumes"] = rows
    subset["selection_policy"] = (
        spec.get("selection_policy", "frozen") + f"; first_{len(rows)}_preregistered"
    )
    output = ARTIFACT_ROOT / f"{anatomy}_{RUN_STAGE}_manifest.json"
    output.write_text(json.dumps(subset, indent=2))
    return subset, output

def verify_checkpoint_manifest(directory, manifest_path):
    directory, manifest_path = Path(directory), Path(manifest_path)
    assert directory.is_dir(), directory
    assert manifest_path.is_file(), manifest_path
    spec = json.loads(manifest_path.read_text())
    assert spec.get("artifacts")
    for row in spec["artifacts"]:
        path = directory / row["relative_path"]
        assert path.is_file(), path
        assert sha256_file(path) == row["sha256"].lower(), path
    return spec

brain_manifest, brain_manifest_path = load_and_verify_volume_manifest(
    MRI_BRAIN_MANIFEST, MRI_BRAIN_DATA, "brain"
)
knee_manifest, knee_manifest_path = load_and_verify_volume_manifest(
    MRI_KNEE_MANIFEST, MRI_KNEE_DATA, "knee"
)
verify_checkpoint_manifest(
    MRI_BRAIN_CHECKPOINT_DIR, MRI_BRAIN_CHECKPOINT_MANIFEST
)
verify_checkpoint_manifest(
    MRI_KNEE_CHECKPOINT_DIR, MRI_KNEE_CHECKPOINT_MANIFEST
)
print(
    "Verified independent volumes:",
    len(brain_manifest["volumes"]),
    "brain /",
    len(knee_manifest["volumes"]),
    "knee",
)
"""
    ),
    markdown(
        r"""
### MRI protocol

- Every selected volume and checkpoint artifact is SHA256-pinned.
- The independent unit is a patient/volume, never a slice.
- All slices are evaluated for pilot and production.
- \(R\in\{4,8,12\}\); \(R=8\) is primary.  ACS and added line counts are
  identical across selectors.
- RaoSense \(K=8,r=2,T=32,J=4\) uses 384 image-equivalent NFE.  The principal
  point-particle control uses \(K=12,T=32\), also 384 NFE.
- Both online reconstructions and canonical single-trajectory replays of the
  final masks are retained in `results.json`.
- fastMRI+ boxes are vertically flipped back to the raw reconstruction
  orientation before ROI evaluation.  Labels are non-exhaustive single-reader
  regions, not diagnostic ground truth.
"""
    ),
    code(
        """
MRI_METHODS = [
    "pigdm_mc",
    "static_match",
    "random_adaptive",
    "equi_adaptive",
    "operator_diag_active",
    "sketch_marginal_active",
    "ens_active",
    "raosense_greedy_active",
    "raosense_h2_active",
    "oracle_active",
]

def mri_command(
    anatomy,
    data_root,
    cohort_manifest,
    checkpoint_dir,
    checkpoint_manifest,
    acceleration,
    active_lines,
    device_id,
):
    cohort = json.loads(Path(cohort_manifest).read_text())
    count = len(cohort["volumes"])
    output = (
        ARTIFACT_ROOT
        / f"mri_{anatomy}_{RUN_STAGE}_R{acceleration}_L{active_lines}"
    )
    command = [
        sys.executable, "scripts/reconstruct_mc.py",
        "--data_path", str(data_root),
        "--volume_manifest", str(cohort_manifest),
        "--dataset_revision", cohort["dataset_revision"],
        "--anatomy", anatomy,
        "--num_volumes", str(count),
        "--minimum_production_volumes", str(max(20, min(count, 100))),
        "--num_slices", "0",
        "--checkpoint_dir", str(checkpoint_dir),
        "--checkpoint_manifest", str(checkpoint_manifest),
        "--adps_root", "external/adps",
        "--expected_adps_commit", ADPS_COMMIT,
        "--acceleration", str(acceleration),
        "--center_fraction", "0.04",
        "--target_resolution", "384", "320",
        "--num_steps", "32",
        "--active_lines", str(active_lines),
        "--active_rounds", "4",
        "--active_after_frac", "0.35",
        "--active_until_frac", "0.80",
        "--ens_particles", "12",
        "--rao_branches", "8",
        "--rao_tangent_rank", "2",
        "--rao_beam_width", "8",
        "--rao_fantasies", "16",
        "--rao_information_samples", "128",
        # Zero disables the engineering prefilter.  H1 bounds cannot safely
        # remove H2 roots because they do not upper-bound future value.
        "--rao_candidate_cap", "0",
        "--methods", *MRI_METHODS,
        "--device", f"cuda:{device_id}",
        "--seed", str(GLOBAL_SEED),
        "--output_dir", str(output),
    ]
    return command, output
"""
    ),
    code(
        """
# Smoke runs are allowed immediately.  Pilot/production require an explicit
# acknowledgement because they can consume multiple GPU-days.
if RUN_STAGE != "smoke":
    assert RUN_EXPENSIVE, "Set RUN_EXPENSIVE=True after reviewing all contracts."

primary_acceleration, primary_lines = 8, 16
brain_command, brain_output = mri_command(
    "brain", MRI_BRAIN_DATA, brain_manifest_path,
    MRI_BRAIN_CHECKPOINT_DIR, MRI_BRAIN_CHECKPOINT_MANIFEST,
    primary_acceleration, primary_lines, 0,
)
knee_device = 1 if GPU_COUNT > 1 else 0
knee_command, knee_output = mri_command(
    "knee", MRI_KNEE_DATA, knee_manifest_path,
    MRI_KNEE_CHECKPOINT_DIR, MRI_KNEE_CHECKPOINT_MANIFEST,
    primary_acceleration, primary_lines, knee_device,
)

if GPU_COUNT > 1:
    jobs = [
        subprocess.Popen(brain_command),
        subprocess.Popen(knee_command),
    ]
    return_codes = [job.wait() for job in jobs]
    assert return_codes == [0, 0], return_codes
else:
    subprocess.run(brain_command, check=True)
    subprocess.run(knee_command, check=True)
"""
    ),
    code(
        """
import pandas as pd

def load_mri_summary(output):
    result = json.loads((Path(output) / "results.json").read_text())
    table = pd.DataFrame(result["summary"]).T
    return result, table

brain_result, brain_table = load_mri_summary(brain_output)
knee_result, knee_table = load_mri_summary(knee_output)
display(brain_table[[
    "selection_psnr_rss_volume_mean",
    "selection_ssim_rss_volume_mean",
    "selection_nmse_rss_volume_mean",
    "nfe_mean", "runtime_s_mean", "peak_cuda_memory_mb_max",
]].dropna(how="all"))
display(knee_table[[
    "selection_psnr_rss_volume_mean",
    "selection_ssim_rss_volume_mean",
    "selection_nmse_rss_volume_mean",
    "nfe_mean", "runtime_s_mean", "peak_cuda_memory_mb_max",
]].dropna(how="all"))

def h2_routing_fraction(result):
    changed, eligible = 0, 0
    for row in result["per_slice"]:
        methods = row["methods"]
        if {
            "raosense_greedy_active",
            "raosense_h2_active",
        }.issubset(methods):
            h1 = methods["raosense_greedy_active"].get(
                "rao_selected_columns", []
            )
            h2 = methods["raosense_h2_active"].get(
                "rao_selected_columns", []
            )
            if h1 and h2:
                eligible += 1
                changed += int(h1[0] != h2[0])
    return changed / max(eligible, 1), eligible

print("Brain H2 root-routing fraction:", h2_routing_fraction(brain_result))
print("Knee H2 root-routing fraction:", h2_routing_fraction(knee_result))
"""
    ),
    markdown(
        r"""
### fastMRI+ pathology evaluation (required before a paper claim)

Use the pinned `external/fastmri-plus/Annotations/{brain,knee}.csv` files.
For each box report PSNR/SSIM/NMSE/HFEN on the pathology region and a
size-matched control patch in the same slice.  Report classwise sensitivity
only for predeclared categories with enough independent subjects.  A frozen
detector must be patient-level cross-fitted on fully sampled references.

The current reconstruction CLI records whole-volume evidence and final masks.
Pathology-box image export/detector evaluation remains
`PENDING_IMPLEMENTATION_AND_COORDINATE_QA`; do not substitute whole-image PSNR
for this clinical endpoint.  First verify the official up/down flip and the
mapping from native RSS support to the model crop on ten overlaid examples.
"""
    ),
    markdown("## 5. KITTI-360 + R2DM programmable-LiDAR arm"),
    markdown(
        r"""
R2DM's \(64\times1024\) image is a derived spherical raster, not a verified
one-to-one record of native HDL-64E firings.  The auditable budget unit here is
therefore an **angular cell** (one raster location carrying range and
reflectance), not a physical ray.  The fixed scout observes 8,192 cells.  Each
\(4\times128\) tile contributes 448 previously unseen cells after scout
overlap, so 16 actions end at exactly 15,360 cells (23.44%).

The active method receives 16 sequential feedback opportunities, scheduled as
four reverse-diffusion checkpoints with four actions each.  A four-tile packet
without within-packet conditioning is a separate deployment ablation.
Confirmatory \(H=2\) evaluates every legal root before its future-value beam.
The retained validity mask is used for evaluation; the current RaoSense
likelihood covers only R2DM's two continuous channels and does not claim a
categorical no-return model.
"""
    ),
    code(
        """
R2DM_ROOT = REPO_DIR / "external/r2dm"
assert KITTI360_DATA_ROOT.is_dir(), (
    "Register for KITTI-360 and set KITTI360_DATA_ROOT."
)
R2DM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
if not R2DM_CHECKPOINT.is_file():
    import urllib.request
    urllib.request.urlretrieve(R2DM_CHECKPOINT_URL, R2DM_CHECKPOINT)
assert sha256_file(R2DM_CHECKPOINT) == R2DM_CHECKPOINT_SHA256

# R2DM uses absolute imports rooted at its checkout.
sys.path.insert(0, str(R2DM_ROOT))
from utils.inference import setup_model, setup_rng

LIDAR_DEVICE = torch.device("cuda:0")
ddpm, lidar_utils, lidar_cfg = setup_model(
    R2DM_CHECKPOINT, device=LIDAR_DEVICE, compile=False
)
assert tuple(lidar_cfg.data.resolution) == (64, 1024)
assert lidar_cfg.diffusion.timestep_type == "continuous"
print(lidar_cfg.data)
"""
    ),
    code(
        """
import datasets as ds
import torch.nn.functional as F

# Load only the official R2DM dataset builder against the licensed raw root.
# The held-out split is drive-separated by the R2DM builder.
original_cwd = Path.cwd()
os.chdir(R2DM_ROOT)
try:
    lidar_dataset = ds.load_dataset(
        path="data/kitti_360",
        name=lidar_cfg.data.projection,
        data_dir=str(KITTI360_DATA_ROOT),
        split=ds.Split.TEST,
        trust_remote_code=True,
    ).with_format("torch")
finally:
    os.chdir(original_cwd)

def preprocess_lidar_item(item):
    depth_metric = item["depth"][None].float().to(LIDAR_DEVICE)
    valid = item["mask"][None].float().to(LIDAR_DEVICE)
    depth = lidar_utils.normalize(
        lidar_utils.convert_depth(depth_metric, mask=valid)
    )
    reflectance = lidar_utils.normalize(
        item["reflectance"][None].float().to(LIDAR_DEVICE)
    )
    scan = torch.cat([depth, reflectance], dim=1)
    scan = valid * scan + (1.0 - valid) * -1.0
    scan = F.interpolate(
        scan, size=lidar_cfg.data.resolution, mode="nearest-exact"
    )
    valid = F.interpolate(
        valid, size=lidar_cfg.data.resolution, mode="nearest-exact"
    ).bool()
    return scan[0], valid[0], int(item["sample_id"])

print("Held-out frames:", len(lidar_dataset))
scan0, valid0, sample0 = preprocess_lidar_item(lidar_dataset[0])
assert scan0.shape == (2, 64, 1024)
print("First sample:", sample0, scan0.min().item(), scan0.max().item())
"""
    ),
    code(
        """
from dataclasses import asdict

from src.samplers.raosense_lidar import (
    build_lidar_tile_actions,
    make_lidar_scout_mask,
    run_raosense_lidar,
)

def apply_actions(mask, actions, selected):
    output = mask.clone()
    flat = output.reshape(-1)
    for action in selected:
        flat[actions[action]] = True
    return output

def fixed_action_ids(actions, count, policy, seed, reference=None):
    ids = sorted(actions)
    if policy == "random":
        generator = np.random.default_rng(seed)
        return generator.choice(ids, size=count, replace=False).tolist()
    if policy == "uniform":
        positions = np.linspace(0, len(ids) - 1, count)
        return [ids[int(round(position))] for position in positions]
    if policy in {"particle_variance", "oracle_error"}:
        assert reference is not None
        scores = {
            action: float(reference.reshape(-1)[indices].mean().item())
            for action, indices in actions.items()
        }
        return sorted(ids, key=lambda action: (-scores[action], action))[:count]
    raise ValueError(policy)

@torch.no_grad()
def canonical_repaint(full_scan, mask, seed):
    known = torch.where(mask, full_scan, -torch.ones_like(full_scan))
    recon = ddpm.repaint(
        known=known[None],
        mask=mask[None].float(),
        num_steps=32,
        num_resample_steps=1,
        jump_length=1,
        rng=setup_rng([seed], device=LIDAR_DEVICE),
    )
    return recon[0].clamp(-1, 1)

def lidar_metrics(reference, reconstruction, valid):
    ref_depth = lidar_utils.revert_depth(
        lidar_utils.denormalize(reference[[0]][None])
    )[0, 0]
    rec_depth = lidar_utils.revert_depth(
        lidar_utils.denormalize(reconstruction[[0]][None])
    )[0, 0]
    valid_pixels = valid[0]
    far = valid_pixels & (ref_depth >= 40.0)
    reflectance_error = (
        lidar_utils.denormalize(reference[[1]])
        - lidar_utils.denormalize(reconstruction[[1]])
    ).abs()[0]
    return {
        "range_mae_m": float(
            (ref_depth[valid_pixels] - rec_depth[valid_pixels]).abs().mean()
        ),
        "far_range_mae_m": float(
            (ref_depth[far] - rec_depth[far]).abs().mean()
        ) if bool(far.any()) else None,
        "reflectance_mae": float(reflectance_error[valid_pixels].mean()),
    }
"""
    ),
    code(
        """
@torch.no_grad()
def evaluate_lidar_frame(dataset_index, active_action_count=16):
    full_scan, valid, sample_id = preprocess_lidar_item(
        lidar_dataset[int(dataset_index)]
    )
    scout = make_lidar_scout_mask(
        channels=2, height=64, width=1024,
        elevation_stride=1, azimuth_stride=8,
        device=LIDAR_DEVICE,
    )
    actions = build_lidar_tile_actions(
        channels=2, height=64, width=1024,
        tile_height=4, tile_width=128,
        observed_mask=scout, device=LIDAR_DEVICE,
    )
    seed = GLOBAL_SEED + int(sample_id)
    rows = []

    scout_recon = canonical_repaint(full_scan, scout, seed)
    oracle_error = (scout_recon - full_scan).square()

    # Endpoint-particle variance control.  This is deliberately expensive and
    # records its selection NFE separately from canonical replay.
    particle_samples = torch.stack([
        canonical_repaint(full_scan, scout, seed + 10_000 + particle)
        for particle in range(12)
    ])
    particle_variance = particle_samples.var(dim=0)

    fixed_specs = {
        "uniform": fixed_action_ids(
            actions, active_action_count, "uniform", seed
        ),
        "random": fixed_action_ids(
            actions, active_action_count, "random", seed
        ),
        "particle_variance_k12": fixed_action_ids(
            actions, active_action_count, "particle_variance", seed,
            reference=particle_variance,
        ),
        "retrospective_oracle": fixed_action_ids(
            actions, active_action_count, "oracle_error", seed,
            reference=oracle_error,
        ),
    }
    for method, selected in fixed_specs.items():
        final_mask = apply_actions(scout, actions, selected)
        queried_angular_cells = int(final_mask.any(dim=0).sum().item())
        assert queried_angular_cells == 15_360, (
            method, queried_angular_cells
        )
        reconstruction = canonical_repaint(
            full_scan, final_mask, seed + 20_000
        )
        rows.append({
            "sample_id": sample_id,
            "dataset_index": int(dataset_index),
            "method": method,
            "selected_actions": [int(value) for value in selected],
            "selection_nfe": 12 * 32 if method == "particle_variance_k12" else 0,
            "canonical_replay_nfe": 32,
            "measured_scalars": int(final_mask.sum().item()),
            "queried_angular_cells": queried_angular_cells,
            **lidar_metrics(full_scan, reconstruction, valid),
        })

    for method, horizon in [
        ("raosense_h1", 1),
        ("raosense_h2", 2),
    ]:
        result = run_raosense_lidar(
            diffusion=ddpm,
            full_scan=full_scan,
            # Headline replay reveals the exact held-out raster encoding.
            # A robustness run must instead pass one once-corrupted tensor;
            # the adapter never clamps hidden truth when this is supplied.
            observed_scan=full_scan,
            initial_mask=scout,
            candidate_actions=actions,
            num_steps=32,
            n_branches=8,
            tangent_rank=2,
            active_actions=active_action_count,
            active_rounds=4,
            active_after_frac=0.30,
            active_until_frac=0.75,
            planning_horizon=horizon,
            beam_width=8,
            num_fantasies=16,
            information_samples=128,
            # Confirmatory H2 evaluates every legal root.  A cap is only a
            # separately labeled engineering ablation.
            planner_candidate_cap=None,
            measurement_noise_variance=1e-3,
            seed=seed,
        )
        canonical = canonical_repaint(
            full_scan, result["mask_final"], seed + 20_000
        )
        queried_angular_cells = int(
            result["mask_final"].any(dim=0).sum().item()
        )
        assert queried_angular_cells == 15_360, (
            method, queried_angular_cells
        )
        rows.append({
            "sample_id": sample_id,
            "dataset_index": int(dataset_index),
            "method": method,
            "selected_actions": [
                int(row.action) for row in result["acquisitions"]
            ],
            "expected_future_information": [
                float(row.expected_future_information)
                for row in result["acquisitions"]
            ],
            "selection_nfe": int(result["nfe"]),
            "canonical_replay_nfe": 32,
            "measured_scalars": int(result["mask_final"].sum().item()),
            "queried_angular_cells": queried_angular_cells,
            "effective_sample_size_mean": float(np.mean([
                row.effective_sample_size for row in result["acquisitions"]
            ])),
            **lidar_metrics(full_scan, canonical, valid),
        })
    return rows
"""
    ),
    code(
        """
# Deterministic temporal thinning across the held-out split.  Store both the
# dataset index and public sample_id so the cohort can be reconstructed.
frame_count = min(TARGET_LIDAR_FRAMES[RUN_STAGE], len(lidar_dataset))
frame_indices = np.linspace(
    0, len(lidar_dataset) - 1, frame_count, dtype=np.int64
)
assert len(np.unique(frame_indices)) == frame_count
lidar_manifest = {
    "dataset": "KITTI-360",
    "r2dm_split": "test",
    "r2dm_projection": str(lidar_cfg.data.projection),
    "r2dm_commit": R2DM_COMMIT,
    "checkpoint_sha256": R2DM_CHECKPOINT_SHA256,
    "selection": "deterministic_temporal_thinning",
    "dataset_indices": frame_indices.tolist(),
}
(ARTIFACT_ROOT / "lidar_manifest.json").write_text(
    json.dumps(lidar_manifest, indent=2)
)
print("LiDAR frames:", frame_count)
"""
    ),
    code(
        """
if RUN_STAGE != "smoke":
    assert RUN_EXPENSIVE

lidar_rows = []
for position, dataset_index in enumerate(frame_indices):
    lidar_rows.extend(evaluate_lidar_frame(int(dataset_index)))
    if (position + 1) % 10 == 0 or position + 1 == frame_count:
        print(f"{position + 1}/{frame_count} frames")
        pd.DataFrame(lidar_rows).to_parquet(
            ARTIFACT_ROOT / "lidar_partial.parquet", index=False
        )

lidar_table = pd.DataFrame(lidar_rows)
lidar_table.to_parquet(ARTIFACT_ROOT / "lidar_results.parquet", index=False)
display(lidar_table.groupby("method")[[
    "range_mae_m", "far_range_mae_m", "reflectance_mae",
    "selection_nfe", "queried_angular_cells", "measured_scalars",
]].agg(["mean", "std"]))

pivot = lidar_table.pivot(
    index="sample_id", columns="method", values="range_mae_m"
)
route = lidar_table.pivot(
    index="sample_id", columns="method", values="selected_actions"
)
root_changed = np.mean([
    h1[0] != h2[0]
    for h1, h2 in zip(route["raosense_h1"], route["raosense_h2"])
])
print("LiDAR H2 root-routing fraction:", root_changed)
print(
    "Paired mean H2-H1 range MAE:",
    float((pivot["raosense_h2"] - pivot["raosense_h1"]).mean()),
    "meters",
)
"""
    ),
    markdown(
        r"""
### LiDAR task metrics still required

Range/reflectance errors above are immediately runnable.  Production promotion
also requires:

- RangeNet semantic mIoU against held-out KITTI-360 semantics, not merely
  agreement with a prediction on the full scan;
- BEV occupancy IoU;
- point-to-plane or Chamfer distance with explicit no-return handling;
- far-range and object-centric errors;
- temporally/drive-clustered paired intervals.

Those label alignments are `PENDING_IMPLEMENTATION`.  Do not call RangeNet
agreement “ground-truth mIoU.”
"""
    ),
    markdown("## 6. Joint image/operator uncertainty stress test"),
    code(
        """
# The analytic suite already checks the scalar Schur-information identity.
# Imaging promotion requires low-dimensional, gauge-fixed nuisance stress:
calibration_protocol = {
    "mri": {
        "nuisance": [
            "global coil phase/gain after fixing one reference coil",
            "piecewise-rigid line motion",
        ],
        "metrics": [
            "image quality",
            "nuisance parameter error",
            "known-operator oracle gap recovered",
            "worst-decile failure",
        ],
    },
    "lidar": {
        "nuisance": [
            "elevation zero-offset",
            "range gain/bias",
        ],
        "metrics": [
            "range MAE",
            "geometry parameter error",
            "known-operator oracle gap recovered",
        ],
    },
    "status": "PENDING_END_TO_END_INTEGRATION",
}
(ARTIFACT_ROOT / "calibration_protocol.json").write_text(
    json.dumps(calibration_protocol, indent=2)
)
display(calibration_protocol)
"""
    ),
    markdown("## 7. Promotion report — no automatic paper claims"),
    code(
        """
def mri_effect(table, method, reference):
    return float(
        table.loc[method, "selection_psnr_rss_volume_mean"]
        - table.loc[reference, "selection_psnr_rss_volume_mean"]
    )

promotion = {
    "status": "REVIEW_REQUIRED",
    "run_stage": RUN_STAGE,
    "synthetic_validation_complete": True,
    "brain": {
        "h2_minus_h1_db": mri_effect(
            brain_table, "raosense_h2_active", "raosense_greedy_active"
        ),
        "h2_minus_particle_db": mri_effect(
            brain_table, "raosense_h2_active", "ens_active"
        ),
        "root_routing_fraction": h2_routing_fraction(brain_result)[0],
    },
    "knee": {
        "h2_minus_h1_db": mri_effect(
            knee_table, "raosense_h2_active", "raosense_greedy_active"
        ),
        "h2_minus_particle_db": mri_effect(
            knee_table, "raosense_h2_active", "ens_active"
        ),
        "root_routing_fraction": h2_routing_fraction(knee_result)[0],
    },
    "lidar": {
        "h2_minus_h1_range_mae_m": float(
            (pivot["raosense_h2"] - pivot["raosense_h1"]).mean()
        ),
        "h2_relative_improvement_vs_particle": float(
            (
                pivot["particle_variance_k12"] - pivot["raosense_h2"]
            ).mean()
            / pivot["particle_variance_k12"].mean()
        ),
        "root_routing_fraction": float(root_changed),
    },
    "required_but_pending": [
        "clustered confidence intervals",
        "fastMRI+ ROI and frozen-detector evaluation",
        "LiDAR semantic/BEV/point-cloud metrics",
        "end-to-end operator-nuisance experiments",
        "faithful external ADS/AdaSense/Morssy/IGS/LOUPE integrations",
    ],
    "thresholds": {
        "minimum_root_routing_fraction": 0.20,
        "mri_primary_gain_db": 0.30,
        "lidar_relative_range_mae_gain": 0.10,
        "lidar_semantic_miou_points": 2.0,
    },
}
(ARTIFACT_ROOT / "promotion_report.json").write_text(
    json.dumps(promotion, indent=2)
)
display(promotion)
"""
    ),
    code(
        """
# Export an immutable bundle only after the report has been inspected.
if USE_GOOGLE_DRIVE:
    export_dir = DRIVE_EXPORT / (
        datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        + f"_{REPO_COMMIT[:12]}_{RUN_STAGE}"
    )
    assert not export_dir.exists(), export_dir
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ARTIFACT_ROOT, export_dir)
    manifest = []
    for path in sorted(export_dir.rglob("*")):
        if path.is_file():
            manifest.append({
                "path": str(path.relative_to(export_dir)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    (export_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print("Exported:", export_dir)
else:
    print("Artifacts retained at:", ARTIFACT_ROOT)
"""
    ),
    markdown(
        """
## Reader checklist

Before copying any number into the paper:

1. Confirm the checkout hashes and `status` fields.
2. Confirm final measurement budgets are identical.
3. Use canonical replay for acquisition claims and online metrics for system
   claims.
4. Compare measured NFE, wall time, and memory—not nominal method names.
5. Bootstrap by patient/drive, not by slice/frame.
6. Inspect pathology overlays, LiDAR no-return cases, and worst-decile failures.
7. Keep all unimplemented external baselines and task metrics visibly pending.
8. If a preregistered gate fails, report the failure and simplify the claim.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "accelerator": "GPU",
        "colab": {
            "name": OUTPUT.name,
            "provenance": [],
            "gpuType": "A100",
        },
    },
)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source, filename=f"{OUTPUT.name}:cell")
        cell.execution_count = None
        cell.outputs = []

rendered = nbf.writes(notebook)
lowered = rendered.lower()
assert "pending_gpu_and_licensed_data" in lowered
assert "run_cifar" not in lowered
assert "run_imagenet" not in lowered
for revision in (
    "6f6bee8a319a971a6091628041b812f23466292a",
    "75feb2fad4b53e48791fcb0257f2e53a25340774",
    "ad66e71e86051d0341e096c60ece30a7de6c5b3f",
    "67ed9a68ee33d3197f93893e8125688470c4a577",
):
    assert revision in rendered
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(rendered)
print(f"Wrote {OUTPUT}")
print(f"Cells: {len(cells)}")
print(f"SHA256: {hashlib.sha256(rendered.encode()).hexdigest()}")
