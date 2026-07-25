"""Render paper/pvas_v2/figs/fig_pvmaps_mri.pdf from a REAL single-slice run.

Runs one fastMRI brain slice through run_fakgd_mc_pv with return_diagnostics=True
and plots the actual per-frequency PV maps at three diffusion stages plus the
cumulative acquired-column trace. No synthetic data.

Run: .venv/bin/python scripts/make_pvmap_figure.py [--device mps]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.samplers.schedules import edm_sigma_schedule  # noqa: E402
from src.samplers.mri_forward import create_mask, fft2c, ifft2c  # noqa: E402
from src.samplers.sense import (  # noqa: E402
    MultiCoilSENSE, estimate_noise_per_coil, estimate_sens_maps_lowres,
)
from src.samplers.multicoil_pv import run_fakgd_mc_pv  # noqa: E402
from src.models.edm_loader import EDMDenoiser, load_edm_model  # noqa: E402


def center_crop(x, H, W):
    h, w = x.shape[-2:]
    top, left = (h - H) // 2, (w - W) // 2
    return x[..., top:top + H, left:left + W]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--slice", type=int, default=8)
    p.add_argument("--acceleration", type=int, default=8)
    p.add_argument("--active_lines", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--sigma_max", type=float, default=10.0)
    args = p.parse_args()

    import glob
    import h5py
    device = torch.device(args.device)
    fp = sorted(glob.glob(str(ROOT / "data/brain_val/file_brain_AXT2_*.h5")))[0]
    with h5py.File(fp) as f:
        ks = torch.from_numpy(f["kspace"][args.slice])
    ci = center_crop(ifft2c(ks), 384, 320)
    rss = torch.sqrt((ci.abs() ** 2).sum(0))
    ci = ci / float(rss.max())
    mc_k = fft2c(ci).to(device)
    Nc, H, W = mc_k.shape

    sigc = estimate_noise_per_coil(mc_k).to(device)
    mask = create_mask(W, 0.08, args.acceleration, seed=42).expand(H, -1).to(device)
    sens = estimate_sens_maps_lowres(mc_k, center_fraction=0.08).to(device)
    op = MultiCoilSENSE(mask, sens, sigc).to(device)
    sigi = torch.full((H, W), float(sigc.mean()), device=device)

    net = load_edm_model(str(ROOT / "checkpoints/edm/supervised_R=1"),
                         method="auto", device=str(device))
    den = EDMDenoiser(net, device=str(device))
    sigma = edm_sigma_schedule(args.num_steps, sigma_max=args.sigma_max).to(device)

    print("running diagnostic slice (production flags)...")
    r = run_fakgd_mc_pv(y_mc=mask * mc_k, sense_op=op, sigma_schedule=sigma,
                        denoiser_fn=den, sigma_i_sq_init=sigi, use_pv_gate=True,
                        pv_radial_smooth=False, pv_normalize=False, pv_n_probes=2,
                        refine_sens=False, active_lines=args.active_lines,
                        active_rounds=2, y_mc_full=mc_k, seed=0,
                        return_diagnostics=True)
    d = r["diagnostics"]
    steps = sorted(d["P_maps"].keys())
    picks = [steps[int(f * (len(steps) - 1))] for f in (0.15, 0.5, 0.9)]
    labels = ["early", "mid", "late"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.6, 2.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.35)
    for i, (s, lab) in enumerate(zip(picks, labels)):
        ax = fig.add_subplot(gs[0, i])
        P = np.fft.fftshift(np.log10(np.abs(d["P_maps"][s]) + 1e-12))
        ax.imshow(P, cmap="magma", aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"step {s}/{args.num_steps} ({lab})", fontsize=8)
        if i == 0:
            ax.set_ylabel(r"$\log_{10}\hat P_i(t)$", fontsize=8)
    ax = fig.add_subplot(gs[0, 3])
    nl = d["num_lines_per_step"]
    ax.step(range(len(nl)), nl, where="post", lw=1.4)
    ax.set_xlabel("diffusion step", fontsize=8)
    ax.set_ylabel("# acquired columns", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.set_title(f"R={args.acceleration}, L={args.active_lines}", fontsize=8)
    fig.tight_layout()
    out = ROOT / "paper/pvas_v2/figs/fig_pvmaps_mri.pdf"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
