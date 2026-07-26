"""Official OpenAI guided-diffusion adapter and fail-closed image loader."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


class GuidedDiffusionUNetAdapter(torch.nn.Module):
    """Expose an OpenAI guided-diffusion model through the diffusers API."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor):
        output = self.model(sample, timestep)
        # Learned-sigma checkpoints concatenate epsilon and variance channels.
        epsilon = output[:, : sample.shape[1]].float()
        return SimpleNamespace(sample=epsilon)


def load_imagenet256_unconditional_adm(
    checkpoint: str | Path,
    guided_diffusion_repo: str | Path,
    *,
    device: str,
    use_fp16: bool = True,
) -> tuple[GuidedDiffusionUNetAdapter, torch.Tensor, dict]:
    """Load OpenAI's official unconditional ImageNet-256 ADM checkpoint."""
    checkpoint = Path(checkpoint).resolve()
    repo = Path(guided_diffusion_repo).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing ADM checkpoint: {checkpoint}")
    package = repo / "guided_diffusion"
    if not package.is_dir():
        raise FileNotFoundError(
            f"missing official guided-diffusion checkout: {package}"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from guided_diffusion.script_util import (  # type: ignore
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    config = model_and_diffusion_defaults()
    config.update({
        "attention_resolutions": "32,16,8",
        "class_cond": False,
        "diffusion_steps": 1000,
        "dropout": 0.0,
        "image_size": 256,
        "learn_sigma": True,
        "noise_schedule": "linear",
        "num_channels": 256,
        "num_head_channels": 64,
        "num_heads": -1,
        "num_res_blocks": 2,
        "resblock_updown": True,
        "rescale_learned_sigmas": False,
        "rescale_timesteps": False,
        "timestep_respacing": "",
        "use_checkpoint": False,
        "use_fp16": bool(use_fp16),
        "use_kl": False,
        "use_new_attention_order": False,
        "use_scale_shift_norm": True,
    })
    model, diffusion = create_model_and_diffusion(**config)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    if use_fp16:
        model.convert_to_fp16()
    adapter = GuidedDiffusionUNetAdapter(model).to(device).eval()
    abar = torch.as_tensor(
        diffusion.alphas_cumprod, dtype=torch.float32, device=device
    )
    provenance = {
        "model": "OpenAI guided-diffusion unconditional ImageNet-256",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "source_repo": "https://github.com/openai/guided-diffusion",
        "source_commit": _git_revision(repo),
        "config": config,
    }
    return adapter, abar, provenance


def _center_crop_arr(image: Image.Image, image_size: int) -> np.ndarray:
    """OpenAI guided-diffusion's deterministic center-crop preprocessing."""
    while min(*image.size) >= 2 * image_size:
        image = image.resize(
            tuple(x // 2 for x in image.size),
            resample=Image.Resampling.BOX,
        )
    scale = image_size / min(*image.size)
    image = image.resize(
        tuple(round(x * scale) for x in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image.convert("RGB"))
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return array[
        crop_y: crop_y + image_size,
        crop_x: crop_x + image_size,
    ]


def load_heldout_image_folder(
    root: str | Path,
    *,
    n_images: int,
    seed: int,
    image_size: int = 256,
) -> tuple[torch.Tensor, list[dict]]:
    """Load an explicit held-out folder and record immutable sample hashes.

    The function fails closed: it never downloads data, changes splits, or
    substitutes another dataset. For ImageNet, point ``root`` at the official
    validation-image directory, not the training directory.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"held-out image directory not found: {root}")
    extensions = {".jpg", ".jpeg", ".png"}
    files = sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=lambda path: path.name,
    )
    if len(files) < n_images:
        raise ValueError(
            f"requested {n_images} images but found only {len(files)} in {root}"
        )
    order = torch.randperm(
        len(files), generator=torch.Generator().manual_seed(int(seed))
    )[:n_images].tolist()
    selected = [files[index] for index in order]
    arrays = []
    manifest = []
    for path in selected:
        with Image.open(path) as image:
            arrays.append(_center_crop_arr(image, image_size))
        manifest.append({
            "relative_path": str(path.relative_to(root)),
            "sha256": _sha256(path),
        })
    tensor = (
        torch.from_numpy(np.stack(arrays))
        .permute(0, 3, 1, 2)
        .float()
        .div(127.5)
        .sub(1.0)
    )
    return tensor, manifest


def validate_imagenet_validation_layout(
    root: str | Path,
    *,
    expected_count: int = 50_000,
) -> dict:
    """Fail closed unless the canonical ILSVRC2012 validation names exist.

    Class-folder reorganization is allowed because basenames are preserved,
    but subsetting, training images, renamed files, and duplicate basenames are
    rejected. Per-file hashes of the sampled subset are still recorded by
    :func:`load_heldout_image_folder`.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory not found: {root}")
    actual = [
        path.name
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    expected = {
        f"ILSVRC2012_val_{index:08d}.JPEG"
        for index in range(1, int(expected_count) + 1)
    }
    actual_set = set(actual)
    if len(actual) != int(expected_count) or actual_set != expected:
        missing = sorted(expected - actual_set)[:5]
        unexpected = sorted(actual_set - expected)[:5]
        raise ValueError(
            "not a canonical ILSVRC2012 validation layout: "
            f"expected {expected_count} unique canonical basenames, "
            f"found {len(actual)} files/{len(actual_set)} unique names; "
            f"missing examples={missing}, unexpected examples={unexpected}"
        )
    name_digest = hashlib.sha256(
        "\n".join(sorted(actual_set)).encode("utf-8")
    ).hexdigest()
    return {
        "dataset": "ILSVRC2012 ImageNet validation",
        "expected_count": int(expected_count),
        "canonical_basenames_verified": True,
        "sorted_basename_manifest_sha256": name_digest,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"
