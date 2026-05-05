"""Load a pre-trained EDM model from the ADPS checkpoint format.

Supports two modes:
  1. 'edm' — the pickle contains the full nn.Module (supervised EDM on clean data).
  2. 'ambient' — the pickle contains only state_dict weights; model is
     reconstructed from training_options.json.

Usage:
    from src.models.edm_loader import load_edm_model, EDMDenoiser

    net = load_edm_model("checkpoints/edm_R1/", method="edm", device="cpu")
    denoiser = EDMDenoiser(net, device="cpu")

    # Use as denoiser_fn for samplers:
    mu_theta = denoiser(x_t_complex, sigma_t)  # complex [H,W] → complex [H,W]
"""

import sys
import os
import pickle
import json
from collections import OrderedDict
from pathlib import Path

import torch
import numpy as np


def load_edm_model(
    model_dir: str,
    method: str = "auto",
    device: str = "cpu",
    adps_root: str | None = None,
) -> torch.nn.Module:
    """Load an EDM network from the ADPS checkpoint directory.

    Args:
        model_dir: Directory containing network-snapshot.pkl (and training_options.json for ambient).
        method: 'edm' (full-model pickle), 'ambient' (state-dict pickle), or 'auto' (detect).
        device: Target device.
        adps_root: Path to the cloned ADPS repo (for dnnlib/torch_utils imports).
            Defaults to <project_root>/external/adps.

    Returns:
        nn.Module with forward(x, sigma) → denoised.
    """
    if adps_root is None:
        adps_root = str(Path(__file__).resolve().parents[2] / "external" / "adps")

    # Auto-clone ADPS if missing so the EDM pickle (which references
    # nvlabs's `dnnlib` and `torch_utils` packages) can resolve its classes.
    if not (Path(adps_root) / "torch_utils").is_dir() or not (Path(adps_root) / "dnnlib").is_dir():
        import subprocess
        Path(adps_root).parent.mkdir(parents=True, exist_ok=True)
        if Path(adps_root).exists():
            import shutil
            shutil.rmtree(adps_root)
        print(f"[edm_loader] cloning ADPS into {adps_root} ...", flush=True)
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/utcsilab/ambient-diffusion-mri.git",
            adps_root,
        ])

    # Add ADPS repo to path so pickle can resolve dnnlib / torch_utils classes
    if adps_root not in sys.path:
        sys.path.insert(0, adps_root)

    snapshot = os.path.join(model_dir, "network-snapshot.pkl")

    if method == "auto":
        # Auto-detect: if training_options.json exists, check if pickle has
        # full model objects or just state_dict
        with open(snapshot, "rb") as f:
            obj = pickle.load(f)
        ema = obj["ema"]
        if isinstance(ema, torch.nn.Module):
            method = "edm"
        else:
            method = "ambient"

    if method == "edm":
        with open(snapshot, "rb") as f:
            net = pickle.load(f)["ema"]
        net = net.to(device).eval()
    elif method == "ambient":
        opts_path = os.path.join(model_dir, "training_options.json")
        with open(opts_path, "r") as f:
            training_options = json.load(f)

        import dnnlib  # requires ADPS on sys.path

        network_kwargs = training_options["network_kwargs"]
        img_channels = 2  # complex MRI: real + imag
        resolution = training_options["dataset_kwargs"]["resolution"]
        interface_kwargs = dict(
            img_resolution=resolution,
            label_dim=0,
            img_channels=img_channels,
        )
        net = dnnlib.util.construct_class_by_name(
            **network_kwargs, **interface_kwargs
        )
        with open(snapshot, "rb") as f:
            state_dict = pickle.load(f)["ema"]
        # Strip torch.compile prefixes
        state_dict = OrderedDict(
            {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        )
        net.load_state_dict(state_dict)
        net = net.to(device).eval()
    else:
        raise ValueError(f"Unknown method: {method}")

    return net


class EDMDenoiser:
    """Wrap an EDM network so it can be used as a denoiser_fn for our samplers.

    Our samplers work with complex-valued images [H, W].
    The EDM network expects real-valued [B, 2, H, W] (channel 0 = real, 1 = imag).
    This wrapper handles the conversion.
    """

    def __init__(self, net: torch.nn.Module, device: str = "cpu"):
        self.net = net
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, x_complex: torch.Tensor, sigma_t: float) -> torch.Tensor:
        """Denoise a complex image at noise level sigma_t.

        Args:
            x_complex: Complex-valued image, shape [H, W].
            sigma_t: Current noise level (scalar).

        Returns:
            Denoised complex image, shape [H, W].
        """
        # Complex [H, W] → real [1, 2, H, W]
        x_real = torch.stack([x_complex.real, x_complex.imag], dim=0)
        x_real = x_real.unsqueeze(0).float().to(self.device)

        sigma = torch.tensor([sigma_t], dtype=torch.float32, device=self.device)

        denoised = self.net(x_real, sigma)  # [1, 2, H, W]

        # Real [1, 2, H, W] → complex [H, W]
        out = denoised[0, 0] + 1j * denoised[0, 1]
        return out.to(x_complex.device)


class OracleDenoiser:
    """Oracle denoiser for testing: μ_θ = x_0 + η·σ_t·noise.

    Requires access to the ground truth x_0.
    """

    def __init__(self, x_gt: torch.Tensor, eta: float = 0.1):
        self.x_gt = x_gt
        self.eta = eta

    def __call__(self, x_t: torch.Tensor, sigma_t: float) -> torch.Tensor:
        noise = (
            torch.randn_like(self.x_gt.real) + 1j * torch.randn_like(self.x_gt.real)
        ) / np.sqrt(2)
        return self.x_gt + self.eta * sigma_t * noise
