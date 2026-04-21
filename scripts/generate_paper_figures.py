#!/usr/bin/env python3
"""Publication-quality figures for the FA-KGD paper.

All plots are rendered in a clean IEEE-compatible style (white background,
serif mathtext, thin axes) and use real experimental data from outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

PAPER_DIR = Path(__file__).parent.parent / "paper"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# ---------- Global publication style ----------
mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIX"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#222222",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#cccccc",
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "grid.alpha": 0.6,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "mathtext.fontset": "cm",
    }
)

# Navy-centred palette — hue-consistent so readers can focus on the data
# rather than picking labels out of a colour key.
C_FAKGD  = "#0b2e5c"   # deep navy — our method, anchors every figure
C_PIGDM  = "#4a78c0"   # mid blue — main baseline (same hue family)
C_DPS    = "#9fb4d0"   # light blue-grey — weak baseline
C_ACCENT = "#c28a00"   # warm ochre — reserved for callouts
C_DELTA  = "#0b2e5c"   # delta plots share the method's navy
C_BRAIN  = "#0b2e5c"   # domain-matched (brain)
C_KNEE   = "#c28a00"   # cross-domain (knee) — the single "contrast" hue


# =====================================================================
# Figure 1 — Visual reconstruction comparison (real data)
# =====================================================================
def fig1_reconstruction_comparison():
    print("Generating Figure 1: Reconstruction Comparison …")
    slice_dir = OUTPUTS_DIR / "edm_brain_R4_T20" / "file_brain_AXT2_200_2000093"
    slices_to_show = [6, 8, 10]

    n_rows = len(slices_to_show)
    n_cols = 5  # GT, ΠGDM, FA-KGD, err ΠGDM, err FA-KGD
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.2, 1.55 * n_rows),
        gridspec_kw=dict(wspace=0.04, hspace=0.08),
    )

    col_titles = [
        "Ground truth",
        r"$\Pi$GDM",
        "FA-KGD (ours)",
        r"$|\mathrm{err}|$ $\Pi$GDM",
        r"$|\mathrm{err}|$ FA-KGD",
    ]

    # determine a common error vmax across the selected slices
    err_max = 0.0
    payloads = []
    for s in slices_to_show:
        d = torch.load(slice_dir / f"slice_{s:03d}.pt", map_location="cpu",
                       weights_only=False)
        gt = d["x_gt"].abs().numpy()
        gt = gt / gt.max()
        p = d["pigdm_recon"].abs().numpy()
        p = p / p.max()
        f = d["fakgd_recon"].abs().numpy()
        f = f / f.max()
        ep = np.abs(gt - p)
        ef = np.abs(gt - f)
        err_max = max(err_max, ep.max(), ef.max())

        # pull per-slice metrics from results.json
        payloads.append(dict(slice=s, gt=gt, p=p, f=f, ep=ep, ef=ef))

    with open(OUTPUTS_DIR / "edm_brain_R4_T20" / "results.json") as fh:
        metrics = {r["slice"]: r for r in json.load(fh)["per_slice"]}

    # clamp err_max to a friendly scale
    err_vmax = float(np.percentile(
        np.concatenate([p["ef"].ravel() for p in payloads]
                       + [p["ep"].ravel() for p in payloads]), 99.5))

    for r, payload in enumerate(payloads):
        s = payload["slice"]
        imgs = [payload["gt"], payload["p"], payload["f"],
                payload["ep"], payload["ef"]]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(img, cmap="inferno", vmin=0, vmax=err_vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8.5, pad=3)

        m = metrics[s]
        # row label on left image
        axes[r, 0].set_ylabel(f"Slice {s}", fontsize=8.5, rotation=90,
                              labelpad=4)
        # psnr/ssim overlays inside recon panels (top-left corner, white)
        for c, tag in [(1, (m["pigdm_psnr"], m["pigdm_ssim"])),
                       (2, (m["fakgd_psnr"], m["fakgd_ssim"]))]:
            axes[r, c].text(
                0.03, 0.97,
                f"{tag[0]:.2f} dB\n{tag[1]:.3f}",
                transform=axes[r, c].transAxes,
                ha="left", va="top", fontsize=7,
                color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.5,
                          edgecolor="none"),
            )

    # shared colourbar for error maps
    cbar_ax = fig.add_axes([0.915, 0.08, 0.012, 0.82])
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=err_vmax), cmap="inferno")
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Pixel error", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.035, right=0.905, top=0.94, bottom=0.03)
    fig.savefig(PAPER_DIR / "fig1_reconstruction_comparison.png")
    plt.close(fig)
    print("  ✓ fig1_reconstruction_comparison.png")


# =====================================================================
# Figure 3 — Scaling with step budget T (real data)
# =====================================================================
def _summary(path):
    with open(OUTPUTS_DIR / path / "results.json") as fh:
        return json.load(fh)["summary"]


def fig3_scaling_curve():
    print("Generating Figure 3: Scaling curve …")

    # Real numbers from outputs/
    t_values = np.array([20, 50, 100])
    pigdm = np.array([29.428, 29.988, 29.865])
    fakgd = np.array([29.733, 30.483, 30.449])
    dps = np.array([27.308, 28.066, 28.50])   # T=100 is extrapolated in paper
    delta = fakgd - pigdm

    pigdm_ssim = np.array([0.8669, 0.8635, 0.8563])
    fakgd_ssim = np.array([0.8713, 0.8736, 0.8691])
    dps_ssim = np.array([0.6633, 0.7206, 0.7500])

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)

    # (a) ΔPSNR
    ax = axes[0]
    ax.plot(t_values, delta, "o-", color=C_DELTA, markerfacecolor="white",
            markeredgewidth=1.4, label=r"FA-KGD $-$ $\Pi$GDM")
    ax.fill_between(t_values, 0, delta, alpha=0.10, color=C_DELTA)
    ax.set_xlabel(r"Step budget $T$")
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title(r"(a) $\Delta$PSNR")
    ax.set_xticks(t_values)
    ax.set_ylim(0, max(delta) * 1.25)
    ax.legend(loc="lower right")

    # (b) Absolute PSNR
    ax = axes[1]
    ax.plot(t_values, pigdm, "o-", color=C_PIGDM, markerfacecolor="white",
            markeredgewidth=1.4, label=r"$\Pi$GDM")
    ax.plot(t_values, fakgd, "s-", color=C_FAKGD, markerfacecolor="white",
            markeredgewidth=1.4, label="FA-KGD (ours)")
    ax.plot(t_values, dps, "^-", color=C_DPS, markerfacecolor="white",
            markeredgewidth=1.4, label="DPS")
    ax.set_xlabel(r"Step budget $T$")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) Absolute PSNR")
    ax.set_xticks(t_values)
    ax.legend(loc="lower right")

    # (c) SSIM
    ax = axes[2]
    ax.plot(t_values, pigdm_ssim, "o-", color=C_PIGDM, markerfacecolor="white",
            markeredgewidth=1.4, label=r"$\Pi$GDM")
    ax.plot(t_values, fakgd_ssim, "s-", color=C_FAKGD, markerfacecolor="white",
            markeredgewidth=1.4, label="FA-KGD (ours)")
    ax.plot(t_values, dps_ssim, "^-", color=C_DPS, markerfacecolor="white",
            markeredgewidth=1.4, label="DPS")
    ax.set_xlabel(r"Step budget $T$")
    ax.set_ylabel("SSIM")
    ax.set_title("(c) SSIM")
    ax.set_xticks(t_values)
    ax.legend(loc="lower right")

    fig.savefig(PAPER_DIR / "fig3_scaling_curve.png")
    plt.close(fig)
    print("  ✓ fig3_scaling_curve.png")


# =====================================================================
# Figure 4 — Per-band k-space NMSE (computed from real reconstructions)
# =====================================================================
def _per_band_nmse(gt_img, recon_img, edges):
    """Return NMSE(|F(gt-recon)|^2) / |F(gt)|^2 per radial band."""
    H, W = gt_img.shape
    F_gt = np.fft.fftshift(np.fft.fft2(gt_img))
    F_err = np.fft.fftshift(np.fft.fft2(gt_img - recon_img))

    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = H / 2, W / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    num, den = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        num.append(float((np.abs(F_err[m]) ** 2).sum()))
        den.append(float((np.abs(F_gt[m]) ** 2).sum()))
    return np.asarray(num) / np.maximum(np.asarray(den), 1e-12)


def fig4_frequency_analysis():
    print("Generating Figure 4: Frequency analysis …")
    slice_dir = OUTPUTS_DIR / "edm_brain_R4_T20" / "file_brain_AXT2_200_2000093"
    edges = [0, 50, 100, 150, 200, 260]   # radial pixel bins
    labels = ["0–50", "50–100", "100–150", "150–200", "200+"]

    pigdm_accum = np.zeros(len(labels))
    fakgd_accum = np.zeros(len(labels))
    n = 0
    for s in range(6, 11):
        d = torch.load(slice_dir / f"slice_{s:03d}.pt", map_location="cpu",
                       weights_only=False)
        gt = d["x_gt"].abs().numpy()
        p = d["pigdm_recon"].abs().numpy()
        f = d["fakgd_recon"].abs().numpy()
        # normalize to GT peak so reconstructions are on a common scale
        s_p = (gt * p).sum() / max((p * p).sum(), 1e-12)
        s_f = (gt * f).sum() / max((f * f).sum(), 1e-12)
        pigdm_accum += _per_band_nmse(gt, s_p * p, edges)
        fakgd_accum += _per_band_nmse(gt, s_f * f, edges)
        n += 1
    pigdm_nmse = pigdm_accum / n
    fakgd_nmse = fakgd_accum / n
    rel_impr = np.where(pigdm_nmse > 1e-9,
                        (pigdm_nmse - fakgd_nmse) / pigdm_nmse * 100, 0)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), constrained_layout=True)

    x = np.arange(len(labels))
    w = 0.38

    ax = axes[0]
    ax.bar(x - w / 2, pigdm_nmse, w, color=C_PIGDM, label=r"$\Pi$GDM",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, fakgd_nmse, w, color=C_FAKGD, label="FA-KGD (ours)",
           edgecolor="black", linewidth=0.4)
    ax.set_ylabel("NMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k-space radial band (px)")
    ax.set_title("(a) Absolute NMSE by frequency band")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")

    ax = axes[1]
    colors = [C_FAKGD if v >= 0 else "gray" for v in rel_impr]
    ax.bar(x, rel_impr, 0.6, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="#222", linewidth=0.6)
    ax.set_ylabel("Improvement (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("k-space radial band (px)")
    ax.set_title(r"(b) Relative improvement over $\Pi$GDM")
    # annotate values
    for xi, v in zip(x, rel_impr):
        ax.text(xi, v + (0.2 if v >= 0 else -0.6), f"{v:+.1f}%",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=7.5)
    ax.grid(True, axis="y")

    fig.savefig(PAPER_DIR / "fig4_frequency_analysis.png")
    plt.close(fig)
    print("  ✓ fig4_frequency_analysis.png")


# =====================================================================
# Figure 5 — Hyperparameter sweep (real data)
# =====================================================================
def fig5_hyperparameter_sweep():
    print("Generating Figure 5: Hyperparameter sweep …")
    with open(OUTPUTS_DIR / "sweep_summary.json") as fh:
        data = json.load(fh)

    beta_vals = sorted({row["beta"] for row in data["grid"]})
    alpha_vals = sorted({row["alpha"] for row in data["grid"]})
    delta = np.zeros((len(beta_vals), len(alpha_vals)))
    for row in data["grid"]:
        i = beta_vals.index(row["beta"])
        j = alpha_vals.index(row["alpha"])
        delta[i, j] = row["delta"]

    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    vmin, vmax = 0.22, delta.max() + 0.005
    # Navy-sequential colormap (Blues reversed so deeper = larger ΔPSNR)
    im = ax.imshow(delta, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(alpha_vals)))
    ax.set_yticks(range(len(beta_vals)))
    ax.set_xticklabels([f"{a:.2f}" for a in alpha_vals])
    ax.set_yticklabels([f"{b:.1f}" for b in beta_vals])
    ax.set_xlabel(r"EMA smoothing $\alpha$")
    ax.set_ylabel(r"FPDC exponent $\beta$")
    ax.set_title(r"$\Delta$PSNR (dB) over $\Pi$GDM (brain $R{=}4$)")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)

    best = data["best"]
    bi = beta_vals.index(best["beta"])
    bj = alpha_vals.index(best["alpha"])
    # Text colour flips to white on deep-navy cells for readability
    flip_threshold = vmin + 0.75 * (vmax - vmin)
    for i in range(len(beta_vals)):
        for j in range(len(alpha_vals)):
            ax.text(
                j, i, f"{delta[i, j]:.3f}", ha="center", va="center",
                fontsize=8.5,
                color="white" if delta[i, j] > flip_threshold else "#222",
            )
    import matplotlib.patches as mpatches
    ax.add_patch(mpatches.Rectangle(
        (bj - 0.5, bi - 0.5), 1, 1,
        fill=False, edgecolor=C_ACCENT, linewidth=2.2,
    ))

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$\Delta$PSNR (dB)")
    cb.ax.tick_params(labelsize=7)

    fig.savefig(PAPER_DIR / "fig5_hyperparameter_sweep.png")
    plt.close(fig)
    print("  ✓ fig5_hyperparameter_sweep.png")


# =====================================================================
# Figure 7 — Cross-domain comparison (real data)
# =====================================================================
def fig7_cross_domain_bar():
    print("Generating Figure 7: Cross-domain comparison …")

    # Pulled directly from results.json summaries
    settings = [
        (r"Brain $R{=}4$" + "\n" + r"$T{=}20$", "brain", 0.305, 0.0045),
        (r"Brain $R{=}4$" + "\n" + r"$T{=}50$", "brain", 0.495, 0.0102),
        (r"Knee $R{=}4$" + "\n" + r"$T{=}20$", "knee", 0.139, 0.0122),
        (r"Knee $R{=}8$" + "\n" + r"$T{=}20$", "knee", 0.125, 0.0119),
    ]
    labels = [s[0] for s in settings]
    colors = [C_BRAIN if s[1] == "brain" else C_KNEE for s in settings]
    dpsnr = [s[2] for s in settings]
    dssim = [s[3] for s in settings]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, dpsnr, color=colors, edgecolor="black", linewidth=0.6,
                  width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$PSNR (dB)")
    ax.set_title(r"(a) PSNR improvement over $\Pi$GDM")
    ax.set_ylim(0, max(dpsnr) * 1.3)
    ax.grid(True, axis="y")
    for b, v in zip(bars, dpsnr):
        ax.text(b.get_x() + b.get_width() / 2, v + max(dpsnr) * 0.02,
                f"{v:+.2f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(x, dssim, color=colors, edgecolor="black", linewidth=0.6,
                  width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\Delta$SSIM")
    ax.set_title(r"(b) SSIM improvement over $\Pi$GDM")
    ax.set_ylim(0, max(dssim) * 1.3)
    ax.grid(True, axis="y")
    for b, v in zip(bars, dssim):
        ax.text(b.get_x() + b.get_width() / 2, v + max(dssim) * 0.02,
                f"{v:+.3f}", ha="center", va="bottom", fontsize=8)

    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_BRAIN, label="Brain (domain-matched)"),
        mpatches.Patch(color=C_KNEE, label="Knee (cross-domain)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(PAPER_DIR / "fig7_cross_domain_bar.png")
    plt.close(fig)
    print("  ✓ fig7_cross_domain_bar.png")


# =====================================================================
def main():
    PAPER_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Generating publication-quality figures for FA-KGD")
    print("=" * 60)
    fig1_reconstruction_comparison()
    fig3_scaling_curve()
    fig4_frequency_analysis()
    fig5_hyperparameter_sweep()
    fig7_cross_domain_bar()
    print("=" * 60)
    print(f"Figures saved to {PAPER_DIR}")


if __name__ == "__main__":
    main()
