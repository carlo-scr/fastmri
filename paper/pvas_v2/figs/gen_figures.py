#!/usr/bin/env python
"""Generate PDF figures for the PVAS WACV 2027 paper.

  fig_calibration.pdf  Spearman-vs-oracle and top-k overlap vs extra NFE
                       (log x) for the PV estimator (pv1..pv16) and the
                       ensemble reference (ens4..ens16), SR + inpaint modes.
  fig_budget.pdf       PSNR vs acquisition budget (static/random/pv/oracle)
                       from cifar_sr_b{6,18,24} + cifar_sr_main.
  fig_nfe_cost.pdf     NFE/image bar chart: PVAS vs ensemble-K (measured)
                       vs ADS / AdaSense (cited multipliers, hatched).
  fig_teaser.pdf       1x3 CelebA-HQ panel (GT / PVAS recon with acquired
                       blocks / PV map) from celeba_qual/panel_data.pt,
                       plus an NFE bar (PVAS vs ensemble, cifar_sr_main).
  fig_qual_sr.pdf      rendered externally by poc_natural/qualitative_celeba.py;
                       this script only emits a placeholder if it is absent.
  fig_pvmaps_mri.pdf   rendered externally from the MRI PV-map dumps;
                       this script only emits a placeholder if it is absent.

All measured numbers come from result JSONs under outputs/ ([R-LOCAL]).
The ONLY non-JSON numbers are the CITED trajectory multipliers for the
hatched bars (documented in CITED_FACTORS with sources). Missing inputs
never crash the script: the affected figure is rendered as a PENDING
placeholder (gray box) so LaTeX always compiles, and a warning lists
what was missing. Placeholder PDFs are tracked with a sidecar
.<name>.placeholder marker so a real externally-rendered figure is
never overwritten.

Palette: colorblind-safe categorical palette validated with the dataviz
skill's validate_palette.js (worst adjacent CVD deltaE 47.2 on white);
series identity is additionally encoded by marker shape and line style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # .../fastmri
MISSING: list[str] = []
STATUS: dict[str, str] = {}

# --------------------------------------------------------------------------
# [CITED] trajectory multipliers -- used ONLY for the hatched "cited" bars in
# fig_nfe_cost.pdf, never for any quality number (sources: docs/lit_review.md):
#   ADS runs Np=16 parallel DPS particles => 16x NFE/step  [nolan2025ads]
#   AdaSense draws s complete posterior-sampling runs per round, e.g. 256
#   trajectories/image for faces                            [elata2024adasense]
# --------------------------------------------------------------------------
CITED_FACTORS = {"ads_np": 16, "adasense_runs": 256}

# Fixed entity -> color mapping (kept identical across all figures).
COL = {
    "pv":       "#2a78d6",   # ours: blue
    "ensemble": "#1baf7a",   # ensemble reference: aqua
    "random":   "#eda100",   # random baseline: yellow
    "static":   "#4a3aa7",   # static baseline: violet
    "cited":    "#e34948",   # cited (never re-run): red
    "oracle":   "#52514e",   # reference bound: secondary ink, dashed
    "muted":    "#898781",
    "grid":     "#e1e0d9",
    "ink":      "#0b0b0b",
}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.labelcolor": COL["ink"], "text.color": COL["ink"],
    "grid.color": COL["grid"], "grid.linewidth": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.dpi": 250,
})


def load(rel: str) -> dict | None:
    p = REPO / "outputs" / rel / "results.json"
    if not p.exists():
        MISSING.append(str(p))
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        MISSING.append(f"{p} (unreadable: {e})")
        return None


def save(fig, name: str, status: str) -> None:
    out = HERE / name
    fig.savefig(out, bbox_inches="tight", dpi=250)
    plt.close(fig)
    STATUS[name] = status
    print(f"  wrote {out.name:22s} [{status}]")


def placeholder(name: str, missing: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.set_axis_off()
    msg = "PENDING\nmissing inputs:\n" + "\n".join(
        Path(m).parent.name for m in missing)
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
            color=COL["muted"], transform=ax.transAxes)
    save(fig, name, "PENDING")


# ------------------------------------------------- placeholder machinery ---
def _marker(name: str) -> Path:
    return HERE / f".{name}.placeholder"


def placeholder_box(name: str, missing: list[str] | None = None) -> None:
    """Gray box + centered 'PENDING: <name>' so LaTeX always compiles."""
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.02, 0.04), 0.96, 0.92,
                               transform=ax.transAxes, facecolor="#d4d3cc",
                               edgecolor="#898781", linewidth=0.8))
    ax.text(0.5, 0.55, f"PENDING: {name}", ha="center", va="center",
            fontsize=9, color=COL["ink"], transform=ax.transAxes)
    if missing:
        sub = "missing: " + ", ".join(sorted({Path(m).parent.name or
                                              Path(m).name for m in missing}))
        ax.text(0.5, 0.32, sub, ha="center", va="center", fontsize=6.5,
                color=COL["muted"], transform=ax.transAxes, wrap=True)
    save(fig, name, "PENDING placeholder")
    _marker(name).touch()


def is_our_placeholder(name: str) -> bool:
    pdf, mk = HERE / name, _marker(name)
    return (pdf.exists() and mk.exists()
            and mk.stat().st_mtime >= pdf.stat().st_mtime - 1.0)


def ensure_external_fig(name: str, inputs: list[Path]) -> None:
    """For figures rendered by other scripts: never overwrite a real PDF,
    but guarantee that SOME PDF exists so the paper compiles."""
    pdf = HERE / name
    missing = [str(p) for p in inputs if not p.exists()]
    if pdf.exists() and not is_our_placeholder(name):
        _marker(name).unlink(missing_ok=True)
        STATUS[name] = "real (external, left as-is)"
        print(f"  kept  {name:22s} [real (external, left as-is)]")
        return
    MISSING.extend(missing)
    placeholder_box(name, missing or None)


# ====================================================== fig_calibration ====
def fig_calibration() -> None:
    modes = {"sr": load("poc_natural/calibration_sr"),
             "inpaint": load("poc_natural/calibration_inpaint")}
    loaded = {k: v for k, v in modes.items() if v is not None}
    if not loaded:
        placeholder("fig_calibration.pdf",
                    [str(REPO / "outputs/poc_natural/calibration_sr"),
                     str(REPO / "outputs/poc_natural/calibration_inpaint")])
        return

    mode_style = {"sr": dict(ls="-", marker="o"),
                  "inpaint": dict(ls=(0, (4, 2)), marker="s")}
    fam_color = {"pv": COL["pv"], "ens": COL["ensemble"]}
    fam_label = {"pv": "PV (ours)", "ens": "Ensemble"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.4))

    def series(d: dict, family: str):
        """Sorted (x, y_spearman, yerr, y_overlap) for one signal family."""
        pts = []
        for name, s in d.get("signals", {}).items():
            if not name.startswith(family):
                continue
            if family == "pv" or name == "ens" or name[3:].isdigit():
                pts.append((s["extra_nfe_per_image"],
                            s["spearman_vs_oracle_mean"],
                            s.get("spearman_vs_oracle_std", 0.0),
                            s["topk_overlap_mean"]))
        return sorted(pts)

    chance_drawn = set()
    for mode, d in loaded.items():
        st = mode_style[mode]
        for fam in ("pv", "ens"):
            pts = series(d, fam)
            if not pts:
                continue
            x = [p[0] for p in pts]
            ax1.errorbar(x, [p[1] for p in pts], yerr=[p[2] for p in pts],
                         color=fam_color[fam], lw=1.6, ms=4.5,
                         elinewidth=0.8, capsize=2, alpha=0.95, **st)
            ax2.plot(x, [p[3] for p in pts], color=fam_color[fam],
                     lw=1.6, ms=4.5, **st)
        # chance level for top-k overlap: k random picks among the
        # unobserved blocks (CIFAR-10 images are 32x32; block from config)
        cfg = d.get("config", {})
        if all(k in cfg for k in ("block", "frac0", "topk")):
            n_total = (32 // cfg["block"]) ** 2
            n_obs = max(0, int(round(cfg["frac0"] * n_total)))
            chance = cfg["topk"] / max(1, n_total - n_obs)
            key = round(chance, 4)
            if key not in chance_drawn:
                chance_drawn.add(key)
                ax2.axhline(chance, color=COL["muted"], lw=1.0, ls=":")
                ax2.annotate("chance", (0.98, chance),
                             xycoords=("axes fraction", "data"),
                             ha="right", va="bottom", fontsize=6.5,
                             color=COL["muted"])

    for ax, ylab in ((ax1, r"Spearman $\rho$ vs. oracle error"),
                     (ax2, "top-$k$ overlap with oracle")):
        ax.set_xscale("log")
        ax.set_xlabel("extra NFE per image (log)")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    handles = []
    for fam in ("pv", "ens"):
        for mode in loaded:
            st = mode_style[mode]
            handles.append(plt.Line2D(
                [], [], color=fam_color[fam], lw=1.6, ms=4.5,
                ls=st["ls"], marker=st["marker"],
                label=f"{fam_label[fam]}, {mode.upper() if mode == 'sr' else mode}"))
    ax1.legend(handles=handles, loc="best", handlelength=2.6)

    status = "populated" if len(loaded) == 2 else "partial"
    save(fig, "fig_calibration.pdf", status)


# =========================================================== fig_budget ====
def fig_budget() -> None:
    tags = ["cifar_sr_b6", "cifar_sr_main", "cifar_sr_b18", "cifar_sr_b24"]
    data = [d for d in (load(f"poc_natural/{t}") for t in tags) if d is not None]
    if not data:
        placeholder("fig_budget.pdf",
                    [str(REPO / "outputs/poc_natural" / t) for t in tags])
        return

    policies = [
        ("static", "Static", COL["static"], "-", "^"),
        ("random", "Random", COL["random"], "-", "D"),
        ("pv", "PVAS (ours)", COL["pv"], "-", "o"),
        ("oracle", "Oracle (bound)", COL["oracle"], (0, (4, 2)), "s"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))

    ends = []  # (last_y, label, color) for de-overlapped direct labels
    for pol, lab, color, ls, mk in policies:
        pts = []
        for d in data:
            if pol not in d.get("summary", {}):
                continue
            budget = d["config"].get("budget_blocks")
            per = d.get("per_image", {}).get(pol, {}).get("psnr", [])
            n = max(1, len(per))
            s = d["summary"][pol]
            sem = s["psnr_std"] / (n ** 0.5)
            pts.append((budget, s["psnr_mean"], sem))
        if not pts:
            continue
        pts.sort()
        x, y, e = zip(*pts)
        ax.errorbar(x, y, yerr=e, color=color, lw=1.6, ls=ls, marker=mk,
                    ms=4.2, elinewidth=0.8, capsize=2, label=lab,
                    zorder=3 if pol == "pv" else 2)
        ends.append([x[-1], y[-1], lab, color])

    # selective direct labels at the last point, nudged apart if they collide
    if ends:
        ends.sort(key=lambda r: r[1])
        span = (max(r[1] for r in ends) - min(r[1] for r in ends)) or 1.0
        min_gap = 0.09 * span
        for i in range(1, len(ends)):
            if ends[i][1] - ends[i - 1][1] < min_gap:
                ends[i][1] = ends[i - 1][1] + min_gap
        for xe, ye, lab, color in ends:
            ax.annotate(lab, (xe, ye), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        color=color, va="center")

    ax.set_xlabel("acquisition budget $L$ (blocks)")
    ax.set_ylabel("PSNR (dB)")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    xs = sorted({d["config"].get("budget_blocks") for d in data
                 if d["config"].get("budget_blocks") is not None})
    if xs:
        ax.set_xticks(xs)
        ax.set_xlim(xs[0] - 1.5, xs[-1] * 1.18)  # room for direct labels
    ax.legend(loc="lower right", handlelength=2.4)

    status = "populated" if len(data) == len(tags) else "partial"
    save(fig, "fig_budget.pdf", status)


# ========================================================= fig_nfe_cost ====
# Supplement candidate: fig_nfe_cost.pdf is not \includegraphics'd by the
# main paper; keep generating it (with tables/sr_budget.tex) for the
# supplement.
def fig_nfe_cost() -> None:
    d = load("poc_natural/cifar_sr_main")
    if d is None:
        placeholder("fig_nfe_cost.pdf",
                    [str(REPO / "outputs/poc_natural/cifar_sr_main")])
        return

    cfg, smry = d.get("config", {}), d.get("summary", {})
    T = cfg.get("num_steps")
    k = cfg.get("ensemble_k", "K")

    bars = []  # (label, nfe, color, cited?)
    if "pv" in smry:
        bars.append(("PVAS\n(ours)", smry["pv"]["nfe_per_image"],
                     COL["pv"], False))
    if "ensemble" in smry:
        bars.append((f"Ensemble\n$K$={k}", smry["ensemble"]["nfe_per_image"],
                     COL["ensemble"], False))
    if T is not None:
        bars.append((f"ADS\n$N_p$={CITED_FACTORS['ads_np']} (cited)",
                     CITED_FACTORS["ads_np"] * T, COL["cited"], True))
        bars.append((f"AdaSense\n$s$={CITED_FACTORS['adasense_runs']} (cited)",
                     CITED_FACTORS["adasense_runs"] * T, "#e87ba4", True))
    if not bars:
        placeholder("fig_nfe_cost.pdf",
                    [str(REPO / "outputs/poc_natural/cifar_sr_main")])
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    xs = range(len(bars))
    for i, (lab, nfe, color, cited) in enumerate(bars):
        ax.bar(i, nfe, width=0.62, color=color,
               hatch="///" if cited else None,
               edgecolor="white" if cited else color, linewidth=0.7)
        ax.annotate(f"{nfe:,.0f}", (i, nfe), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color=COL["ink"])
    ax.set_yscale("log")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_ylabel("NFE per image (log)")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")

    # passive-trajectory reference (T denoiser calls, from the same JSON)
    base = smry.get("static", {}).get("nfe_per_image")
    ymin, ymax = ax.get_ylim()
    handles = [
        Patch(facecolor=COL["muted"], label="measured (this repo)"),
        Patch(facecolor=COL["muted"], hatch="///", edgecolor="white",
              label="cited, not re-run"),
    ]
    if base is not None:
        ymin = min(ymin, base * 0.4)  # keep the reference line off the floor
        ax.axhline(base, color=COL["muted"], lw=1.0, ls=":")
        handles.append(plt.Line2D([], [], color=COL["muted"], lw=1.0, ls=":",
                                  label="passive trajectory ($T$)"))
    ax.set_ylim(ymin, ymax * 1.6)  # headroom for value labels on log scale
    ax.legend(handles=handles, loc="upper left")

    save(fig, "fig_nfe_cost.pdf", "populated")


# =========================================================== fig_teaser ====
def fig_teaser() -> None:
    """1x3 CelebA-HQ panel (GT / PVAS recon + acquired blocks / PV map) plus
    an NFE bar (PVAS vs ensemble selection, both read from cifar_sr_main)."""
    name = "fig_teaser.pdf"
    panel_pt = REPO / "outputs/poc_natural/celeba_qual/panel_data.pt"
    if not panel_pt.exists():
        MISSING.append(str(panel_pt))
        placeholder_box(name, [str(panel_pt)])
        return

    try:
        import torch
        try:
            data = torch.load(panel_pt, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(panel_pt, map_location="cpu")
        images = data["images"]
        recon = data["recons"]["pv"]
        mask = data["masks"]["pv"]
        pvmap = data["pvmaps"]["pv"]
    except Exception as e:  # corrupt / partial dump: never crash
        MISSING.append(f"{panel_pt} (unreadable: {e})")
        placeholder_box(name, [f"{panel_pt} ({e})"])
        return

    d_sr = load("poc_natural/cifar_sr_main")   # NFE bar numbers only
    nfe = (d_sr or {}).get("nfe", {})
    if d_sr and not nfe:  # older JSONs: fall back to per-policy summaries
        nfe = {p: s.get("nfe_per_image")
               for p, s in d_sr.get("summary", {}).items()}
    ens_k = (d_sr or {}).get("config", {}).get("ensemble_k", "K")

    def to01(x):
        return ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()

    i = 0  # first face of the qualitative batch
    H = images.shape[-1]
    m = mask[i, 0].float()
    if m.shape[-1] != H:  # block-grid mask -> image-resolution contour
        import torch.nn.functional as F
        m = F.interpolate(m[None, None], size=(H, H), mode="nearest")[0, 0]

    fig = plt.figure(figsize=(6.9, 1.95))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.85],
                          wspace=0.06, left=0.01, right=0.99,
                          top=0.86, bottom=0.02)
    ax_gt, ax_rec, ax_pv = (fig.add_subplot(gs[0, j]) for j in range(3))
    ax_gt.imshow(to01(images[i]))
    ax_gt.set_title("ground truth", fontsize=7.5)
    ax_rec.imshow(to01(recon[i]))
    ax_rec.contour(m.numpy(), levels=[0.5], colors="cyan", linewidths=0.7)
    ax_rec.set_title("PVAS recon + acquired blocks", fontsize=7.5)
    ax_pv.imshow(pvmap[i, 0].numpy(), cmap="magma")
    ax_pv.set_title("posterior-variance map", fontsize=7.5)
    for ax in (ax_gt, ax_rec, ax_pv):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    ax_bar = fig.add_subplot(gs[0, 3])
    bars = [("PVAS\n(ours)", nfe.get("pv"), COL["pv"]),
            (f"Ensemble\n$K$={ens_k}", nfe.get("ensemble"), COL["ensemble"])]
    bars = [b for b in bars if b[1] is not None]
    if bars:
        for j, (lab, v, c) in enumerate(bars):
            ax_bar.bar(j, v, width=0.6, color=c)
            ax_bar.annotate(f"{v:,.0f}", (j, v), xytext=(0, 2),
                            textcoords="offset points", ha="center",
                            va="bottom", fontsize=7, color=COL["ink"])
        ax_bar.set_xticks(range(len(bars)))
        ax_bar.set_xticklabels([b[0] for b in bars], fontsize=6.5)
        ax_bar.set_ylim(0, max(b[1] for b in bars) * 1.22)
        ax_bar.set_ylabel("NFE / image", fontsize=7)
        ax_bar.tick_params(labelsize=6.5)
        ax_bar.set_title("selection cost", fontsize=7.5)
        ax_bar.grid(True, axis="y")
        status = "populated"
    else:
        ax_bar.set_axis_off()
        ax_bar.text(0.5, 0.5, "NFE bar:\nPENDING", ha="center", va="center",
                    fontsize=7, color=COL["muted"], transform=ax_bar.transAxes)
        status = "partial (NFE bar pending)"

    _marker(name).unlink(missing_ok=True)
    save(fig, name, status)


# ================================================================ main ======
def main() -> int:
    print(f"gen_figures: repo = {REPO}")
    fig_calibration()
    fig_budget()
    fig_nfe_cost()
    fig_teaser()
    # Externally rendered figures: emit gray-box placeholders when absent so
    # the paper always compiles; never overwrite a real (non-placeholder) PDF.
    ensure_external_fig(
        "fig_qual_sr.pdf",
        [REPO / "outputs/poc_natural/celeba_qual/panel_data.pt"])
    ensure_external_fig(
        "fig_pvmaps_mri.pdf",
        [REPO / "outputs/PVAS_local/R4_a8/results.json"])

    print("\n-- figure status --")
    for kk, v in STATUS.items():
        print(f"  {kk:22s} {v}")
    if MISSING:
        print("\nWARNING: missing/unreadable inputs:", file=sys.stderr)
        for m in MISSING:
            print(f"  - {m}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
