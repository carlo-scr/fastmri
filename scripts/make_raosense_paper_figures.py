"""Generate deterministic vector figures for the RaoSense paper draft."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "raosense" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

BLUE = "#276FBF"
TEAL = "#16A085"
ORANGE = "#E67E22"
RED = "#C0392B"
PURPLE = "#7D3C98"
DARK = "#263238"
PALE = "#F5F7FA"


def _box(axis, xy, width, height, text, color, fontsize=8):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=DARK,
    )


def _arrow(
    axis,
    start,
    end,
    color=DARK,
    style="-|>",
    connectionstyle="arc3",
):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=1.2,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def method_overview():
    figure, axis = plt.subplots(figsize=(7.05, 2.55))
    figure.patch.set_facecolor("white")
    axis.set_facecolor(PALE)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.02,
        0.94,
        "RaoSense at one acquisition checkpoint",
        fontsize=11,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    _box(
        axis,
        (0.02, 0.49),
        0.15,
        0.27,
        "$K$ guided diffusion\nbranches $x_t^k$",
        BLUE,
    )
    _box(
        axis,
        (0.22, 0.49),
        0.17,
        0.27,
        "Denoised means $m_k$\n+ $r$ local JVPs",
        TEAL,
    )
    _box(
        axis,
        (0.44, 0.49),
        0.18,
        0.27,
        "Operator-pushed mixture\n$\\sum_k w_k\\,\\mathcal{N}(\\mu_k,V_kV_k^\\top)$",
        PURPLE,
        fontsize=7.5,
    )
    _box(
        axis,
        (0.68, 0.49),
        0.14,
        0.27,
        "Sufficient action space\n$d\\leq\\sum r_k+K-1$",
        ORANGE,
        fontsize=7.5,
    )
    _box(
        axis,
        (0.86, 0.49),
        0.12,
        0.27,
        "H=1 / H=2\nacquisition",
        RED,
    )
    for left, right in [
        ((0.17, 0.625), (0.22, 0.625)),
        ((0.39, 0.625), (0.44, 0.625)),
        ((0.62, 0.625), (0.68, 0.625)),
        ((0.82, 0.625), (0.86, 0.625)),
    ]:
        _arrow(axis, left, right)

    _box(
        axis,
        (0.20, 0.10),
        0.25,
        0.19,
        "Actual measurement: analytic branch\nweight, mean, and factor update",
        TEAL,
        fontsize=7.5,
    )
    _box(
        axis,
        (0.55, 0.10),
        0.25,
        0.19,
        "Corrected clean estimates feed the\nnext reverse-diffusion step",
        BLUE,
        fontsize=7.5,
    )
    _arrow(
        axis,
        (0.92, 0.49),
        (0.45, 0.29),
        RED,
        connectionstyle="arc3,rad=-0.07",
    )
    _arrow(axis, (0.45, 0.195), (0.55, 0.195))
    _arrow(
        axis,
        (0.68, 0.10),
        (0.095, 0.49),
        BLUE,
        connectionstyle="arc3,rad=-0.32",
    )

    axis.text(
        0.02,
        0.365,
        "MRI: global complex SENSE lines",
        color=BLUE,
        fontsize=7.5,
        fontweight="bold",
        bbox={"facecolor": PALE, "edgecolor": "none", "pad": 1.0},
    )
    axis.text(
        0.72,
        0.365,
        "LiDAR: local range-image angular tiles",
        color=ORANGE,
        fontsize=7.5,
        fontweight="bold",
        bbox={"facecolor": PALE, "edgecolor": "none", "pad": 1.0},
    )
    figure.tight_layout(pad=0.35)
    figure.savefig(
        OUTPUT / "method_overview.pdf",
        bbox_inches="tight",
        metadata={"Creator": "make_raosense_paper_figures.py"},
    )
    figure.savefig(
        OUTPUT / "method_overview.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def analytic_validation():
    figure, axes = plt.subplots(1, 4, figsize=(7.05, 1.9))
    figure.patch.set_facecolor("white")

    axis = axes[0]
    methods = ["Particles", "RaoSense"]
    values = [0.3745677, 0.1439694]
    axis.bar(methods, values, color=[ORANGE, TEAL], width=0.65)
    axis.set_ylabel("covariance MSE", fontsize=7)
    axis.set_title("Rao--Blackwell risk", fontsize=8, fontweight="bold")
    axis.text(
        0.5,
        0.92,
        "$-61.6\\%$",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=TEAL,
        fontweight="bold",
    )

    axis = axes[1]
    axis.barh(["scalar action", "sufficient"], [4096, 11], color=[BLUE, PURPLE])
    axis.set_xscale("log")
    axis.set_xlabel("dimension (log)", fontsize=7)
    axis.set_title("372$\\times$ compression", fontsize=8, fontweight="bold")

    axis = axes[2]
    axis.vlines(0, 2.8694, 4.2557, color=PURPLE, linewidth=5, alpha=0.35)
    axis.scatter([0], [4.1573], color=RED, zorder=3, s=25, label="estimate")
    axis.scatter(
        [0, 0],
        [2.8694, 4.2557],
        color=PURPLE,
        zorder=3,
        s=18,
        marker="_",
    )
    axis.set_xlim(-0.4, 0.4)
    axis.set_ylim(2.72, 4.48)
    axis.set_xticks([])
    axis.set_ylabel("information (nats)", fontsize=7)
    axis.set_title("Certified sandwich", fontsize=8, fontweight="bold")
    axis.text(0.08, 4.2557, "upper", va="center", fontsize=6.5)
    axis.text(0.08, 2.8694, "lower", va="center", fontsize=6.5)

    axis = axes[3]
    values = [1.0986, 0.1757]
    axis.bar(
        ["treated\nknown", "nuisance\nmodeled"],
        values,
        color=[RED, TEAL],
        width=0.65,
    )
    axis.set_ylabel("target information", fontsize=7)
    axis.set_title("Operator optimism", fontsize=8, fontweight="bold")
    axis.text(
        0.5,
        0.78,
        "6.25$\\times$",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=RED,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
    )

    for axis in axes:
        axis.tick_params(axis="both", labelsize=6.5, length=2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.16, linewidth=0.5)

    figure.tight_layout(w_pad=0.8)
    figure.savefig(
        OUTPUT / "analytic_validation.pdf",
        bbox_inches="tight",
        metadata={"Creator": "make_raosense_paper_figures.py"},
    )
    figure.savefig(
        OUTPUT / "analytic_validation.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


if __name__ == "__main__":
    method_overview()
    analytic_validation()
    for output in sorted(OUTPUT.glob("*")):
        print(output.relative_to(ROOT))
