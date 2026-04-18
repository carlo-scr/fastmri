"""
Generate publication-ready figures for FA-KGD paper.
Navy blue color scheme, IEEE/NeurIPS style.

Usage:
    cd <project_root>
    python notebooks/visualizations/generate_paper_figures.py

Outputs saved to notebooks/visualizations/figures/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from skimage.metrics import structural_similarity as ssim_fn
from scipy.ndimage import uniform_filter

from src.samplers.mri_forward import fft2c, build_radius_grid

# ─── Style ────────────────────────────────────────────────────
# Navy-blue palette for IEEE / NeurIPS
NAVY       = '#0A1628'
DARK_BLUE  = '#1B2A4A'
MID_BLUE   = '#2E5090'
ACCENT     = '#4A90D9'
LIGHT_BLUE = '#7EB8E0'
SLATE      = '#94A3B8'
CORAL      = '#E8604C'    # contrast accent for FA-KGD
GOLD       = '#D4A843'    # optional highlight
WHITE      = '#FFFFFF'
LIGHT_GRAY = '#F1F5F9'

# Method colors
COLOR_PIGDM = MID_BLUE
COLOR_FAKGD = CORAL
COLOR_GT    = NAVY

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.6,
    'axes.edgecolor': DARK_BLUE,
    'axes.labelcolor': NAVY,
    'xtick.color': DARK_BLUE,
    'ytick.color': DARK_BLUE,
    'grid.color': '#CBD5E1',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'text.color': NAVY,
    'lines.linewidth': 1.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.prop_cycle': matplotlib.cycler(color=[COLOR_PIGDM, COLOR_FAKGD, GOLD, SLATE]),
})

OUTPUTS = PROJECT_ROOT / 'outputs'
FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────

def load_results(run_name):
    run_dir = OUTPUTS / run_name
    with open(run_dir / 'results.json') as f:
        meta = json.load(f)
    slices = []
    for vol_dir in sorted(run_dir.iterdir()):
        if not vol_dir.is_dir():
            continue
        for pt_file in sorted(vol_dir.glob('*.pt')):
            slices.append(torch.load(pt_file, map_location='cpu', weights_only=False))
    return meta, slices


def compute_psnr(gt, recon):
    gt_m, r_m = gt.abs(), recon.abs()
    mse = ((gt_m - r_m) ** 2).mean()
    if mse < 1e-12:
        return 100.0
    return (10 * torch.log10(gt_m.max()**2 / mse)).item()


def compute_ssim(gt, recon):
    g, r = gt.abs().numpy(), recon.abs().numpy()
    return ssim_fn(g, r, data_range=g.max() - g.min())


def _add_panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=NAVY, va='top')


# ═════════════════════════════════════════════════════════════
# Figure 1 — Reconstruction comparison grid (GT / ΠGDM / FA-KGD / error)
# ═════════════════════════════════════════════════════════════

def fig1_reconstruction_comparison():
    print('[Fig 1] Reconstruction comparison grid …')
    # Use T=50 results for best quality
    run_name = 'edm_brain_R4_T50'
    if not (OUTPUTS / run_name / 'results.json').exists():
        run_name = 'edm_brain_R4'
    _, slices_data = load_results(run_name)
    n_show = min(3, len(slices_data))

    fig, axes = plt.subplots(n_show, 5, figsize=(7.16, 1.9 * n_show),  # IEEE column width
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.08})
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        d = slices_data[i]
        gt    = d['x_gt'].abs()
        pigdm = d['pigdm_recon'].abs()
        fakgd = d['fakgd_recon'].abs()

        psnr_p = compute_psnr(d['x_gt'], d['pigdm_recon'])
        psnr_f = compute_psnr(d['x_gt'], d['fakgd_recon'])
        ssim_p = compute_ssim(d['x_gt'], d['pigdm_recon'])
        ssim_f = compute_ssim(d['x_gt'], d['fakgd_recon'])

        err_p = (gt - pigdm).abs()
        err_f = (gt - fakgd).abs()
        vmax  = gt.max().item()
        emax  = max(err_p.max().item(), err_f.max().item()) * 0.8  # slight boost

        # GT
        axes[i, 0].imshow(gt.numpy(), cmap='gray', vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 0].set_title('Ground Truth', fontweight='bold', pad=4)

        # ΠGDM
        axes[i, 1].imshow(pigdm.numpy(), cmap='gray', vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 1].set_title(r'$\Pi$GDM', fontweight='bold', pad=4)
        axes[i, 1].text(0.03, 0.04, f'{psnr_p:.2f} dB\n{ssim_p:.4f}',
                        transform=axes[i, 1].transAxes, fontsize=6.5,
                        color=WHITE, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc=MID_BLUE, alpha=0.85, lw=0))

        # FA-KGD
        axes[i, 2].imshow(fakgd.numpy(), cmap='gray', vmin=0, vmax=vmax)
        if i == 0:
            axes[i, 2].set_title('FA-KGD (ours)', fontweight='bold', pad=4, color=CORAL)
        axes[i, 2].text(0.03, 0.04, f'{psnr_f:.2f} dB\n{ssim_f:.4f}',
                        transform=axes[i, 2].transAxes, fontsize=6.5,
                        color=WHITE, va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc=CORAL, alpha=0.85, lw=0))

        # Error ΠGDM
        axes[i, 3].imshow(err_p.numpy(), cmap='hot', vmin=0, vmax=emax)
        if i == 0:
            axes[i, 3].set_title(r'|Error| $\Pi$GDM', fontweight='bold', pad=4)

        # Error FA-KGD
        axes[i, 4].imshow(err_f.numpy(), cmap='hot', vmin=0, vmax=emax)
        if i == 0:
            axes[i, 4].set_title('|Error| FA-KGD', fontweight='bold', pad=4)

        for j in range(5):
            axes[i, j].axis('off')

    fig.savefig(FIG_DIR / 'fig1_reconstruction_comparison.pdf')
    fig.savefig(FIG_DIR / 'fig1_reconstruction_comparison.png')
    plt.close(fig)
    print('  ✓ Saved fig1_reconstruction_comparison.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 2 — Zoomed-in crops with error difference
# ═════════════════════════════════════════════════════════════

def fig2_zoomed_crops():
    print('[Fig 2] Zoomed crops …')
    run_name = 'edm_brain_R4_T50'
    if not (OUTPUTS / run_name / 'results.json').exists():
        run_name = 'edm_brain_R4'
    _, slices_data = load_results(run_name)

    d = slices_data[0]  # best slice
    gt    = d['x_gt'].abs()
    pigdm = d['pigdm_recon'].abs()
    fakgd = d['fakgd_recon'].abs()
    H, W  = gt.shape
    crop  = 80

    # Auto-pick 2 interesting regions
    gt_np = gt.numpy()
    lv = uniform_filter(gt_np**2, size=crop) - uniform_filter(gt_np, size=crop)**2
    lv[:crop//2, :] = lv[-crop//2:, :] = 0
    lv[:, :crop//2] = lv[:, -crop//2:] = 0
    regions = []
    for _ in range(2):
        idx = np.unravel_index(lv.argmax(), lv.shape)
        regions.append((idx[0] - crop//2, idx[1] - crop//2))
        r0, c0 = max(0, idx[0]-crop), max(0, idx[1]-crop)
        r1, c1 = min(H, idx[0]+crop), min(W, idx[1]+crop)
        lv[r0:r1, c0:c1] = 0

    fig = plt.figure(figsize=(7.16, 3.8))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[1.0, 2.2], wspace=0.12)

    # Left: full image with crop boxes
    ax_full = fig.add_subplot(gs_main[0])
    ax_full.imshow(gt_np, cmap='gray')
    box_colors = [ACCENT, CORAL]
    for j, (r, c) in enumerate(regions):
        rect = Rectangle((c, r), crop, crop, lw=1.8,
                          edgecolor=box_colors[j], facecolor='none', linestyle='-')
        ax_full.add_patch(rect)
    ax_full.set_title('Ground Truth', fontweight='bold', fontsize=9, pad=4)
    ax_full.axis('off')
    _add_panel_label(ax_full, '(a)', x=-0.05)

    # Right: crop grid
    gs_right = gridspec.GridSpecFromSubplotSpec(len(regions), 4, subplot_spec=gs_main[1],
                                                wspace=0.04, hspace=0.06)
    col_titles = ['Ground Truth', r'$\Pi$GDM', 'FA-KGD (ours)', 'Error diff']

    for j, (r, c) in enumerate(regions):
        gt_c = gt[r:r+crop, c:c+crop]
        pi_c = pigdm[r:r+crop, c:c+crop]
        fa_c = fakgd[r:r+crop, c:c+crop]
        err_p = (gt_c - pi_c).abs()
        err_f = (gt_c - fa_c).abs()
        err_diff = err_p - err_f  # positive = FA-KGD better
        vmax_c = gt_c.max().item()
        lim = max(abs(err_diff.min().item()), abs(err_diff.max().item()))

        imgs = [gt_c.numpy(), pi_c.numpy(), fa_c.numpy(), err_diff.numpy()]
        cmaps = ['gray', 'gray', 'gray', 'RdBu']
        vmins = [0, 0, 0, -lim]
        vmaxs = [vmax_c, vmax_c, vmax_c, lim]

        for k in range(4):
            ax = fig.add_subplot(gs_right[j, k])
            ax.imshow(imgs[k], cmap=cmaps[k], vmin=vmins[k], vmax=vmaxs[k],
                      interpolation='nearest')
            ax.axis('off')
            if j == 0:
                color = CORAL if k == 2 else NAVY
                ax.set_title(col_titles[k], fontsize=8, fontweight='bold', pad=3, color=color)
            # Colored border matching box
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(box_colors[j])
                spine.set_linewidth(1.2)

    # Panel (b) label on first crop row
    crop_ax0 = fig.axes[-4]  # first axis of the top crop row
    _add_panel_label(crop_ax0, '(b)', x=-0.18)

    fig.savefig(FIG_DIR / 'fig2_zoomed_crops.pdf')
    fig.savefig(FIG_DIR / 'fig2_zoomed_crops.png')
    plt.close(fig)
    print('  ✓ Saved fig2_zoomed_crops.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 3 — Scaling curve (Δ PSNR & SSIM vs T)
# ═════════════════════════════════════════════════════════════

def fig3_scaling_curve():
    print('[Fig 3] Scaling curve …')
    candidates = {
        20:  ['edm_brain_R4_T20', 'edm_brain_R4'],
        50:  ['edm_brain_R4_T50_ssim', 'edm_brain_R4_T50'],
        100: ['edm_brain_R4_T100'],
    }

    T_vals, d_psnr, d_psnr_std = [], [], []
    p_psnr, f_psnr = [], []
    p_ssim, f_ssim, d_ssim = [], [], []

    for T in sorted(candidates):
        run_name = None
        for name in candidates[T]:
            if (OUTPUTS / name / 'results.json').exists():
                run_name = name
                break
        if run_name is None:
            continue

        with open(OUTPUTS / run_name / 'results.json') as fh:
            s = json.load(fh)['summary']

        T_vals.append(T)
        d_psnr.append(s['delta_psnr_mean'])
        d_psnr_std.append(s['delta_psnr_std'])
        p_psnr.append(s['pigdm_mean_psnr'])
        f_psnr.append(s['fakgd_mean_psnr'])

        # SSIM
        if 'pigdm_mean_ssim' in s:
            p_ssim.append(s['pigdm_mean_ssim'])
            f_ssim.append(s['fakgd_mean_ssim'])
            d_ssim.append(s['delta_ssim_mean'])
        else:
            _, sl = load_results(run_name)
            ps = [compute_ssim(d['x_gt'], d['pigdm_recon']) for d in sl]
            fs = [compute_ssim(d['x_gt'], d['fakgd_recon']) for d in sl]
            p_ssim.append(np.mean(ps))
            f_ssim.append(np.mean(fs))
            d_ssim.append(np.mean(fs) - np.mean(ps))

    if len(T_vals) < 2:
        print('  ⚠ Need ≥2 T values, skipping.')
        return

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.4))

    # (a) Δ PSNR vs T
    ax1.errorbar(T_vals, d_psnr, yerr=d_psnr_std, fmt='o-', color=CORAL,
                 markeredgecolor=NAVY, markeredgewidth=0.5, markersize=7,
                 capsize=4, capthick=1.0, elinewidth=1.0, zorder=5)
    ax1.fill_between(T_vals,
                     [v - s for v, s in zip(d_psnr, d_psnr_std)],
                     [v + s for v, s in zip(d_psnr, d_psnr_std)],
                     color=CORAL, alpha=0.12)
    ax1.set_xlabel('Diffusion steps $T$')
    ax1.set_ylabel('$\\Delta$ PSNR (dB)')
    ax1.set_xticks(T_vals)
    ax1.grid(True)
    _add_panel_label(ax1, '(a)')

    # (b) Absolute PSNR vs T
    ax2.plot(T_vals, p_psnr, 'o-', color=COLOR_PIGDM, markersize=6,
             markeredgecolor=NAVY, markeredgewidth=0.5, label=r'$\Pi$GDM')
    ax2.plot(T_vals, f_psnr, 's-', color=COLOR_FAKGD, markersize=6,
             markeredgecolor=NAVY, markeredgewidth=0.5, label='FA-KGD')
    ax2.set_xlabel('Diffusion steps $T$')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_xticks(T_vals)
    ax2.legend(frameon=True, fancybox=False, edgecolor=SLATE, framealpha=0.9)
    ax2.grid(True)
    _add_panel_label(ax2, '(b)')

    # (c) SSIM vs T
    ax3.plot(T_vals, p_ssim, 'o-', color=COLOR_PIGDM, markersize=6,
             markeredgecolor=NAVY, markeredgewidth=0.5, label=r'$\Pi$GDM')
    ax3.plot(T_vals, f_ssim, 's-', color=COLOR_FAKGD, markersize=6,
             markeredgecolor=NAVY, markeredgewidth=0.5, label='FA-KGD')
    ax3.set_xlabel('Diffusion steps $T$')
    ax3.set_ylabel('SSIM')
    ax3.set_xticks(T_vals)
    ax3.legend(frameon=True, fancybox=False, edgecolor=SLATE, framealpha=0.9)
    ax3.grid(True)
    _add_panel_label(ax3, '(c)')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_scaling_curve.pdf')
    fig.savefig(FIG_DIR / 'fig3_scaling_curve.png')
    plt.close(fig)
    print('  ✓ Saved fig3_scaling_curve.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 4 — Frequency-band NMSE analysis
# ═════════════════════════════════════════════════════════════

def fig4_frequency_analysis():
    print('[Fig 4] Frequency-band analysis …')
    run_name = 'edm_brain_R4_T50'
    if not (OUTPUTS / run_name / 'results.json').exists():
        run_name = 'edm_brain_R4'
    _, slices_data = load_results(run_name)

    n_bands = 5
    pigdm_b = np.zeros(n_bands)
    fakgd_b = np.zeros(n_bands)
    n = len(slices_data)

    for d in slices_data:
        gt, pi, fa = d['x_gt'], d['pigdm_recon'], d['fakgd_recon']
        H, W = gt.shape[-2], gt.shape[-1]
        rg = build_radius_grid(H, W)
        r_max = rg.max().item()
        edges = np.linspace(0, r_max, n_bands + 1)
        for method, arr in [(pi, pigdm_b), (fa, fakgd_b)]:
            gt_k = fft2c(gt)
            m_k  = fft2c(method)
            err  = (gt_k - m_k).abs()**2
            pwr  = gt_k.abs()**2
            for b in range(n_bands):
                ring = (rg >= edges[b]) & (rg < edges[b+1])
                if ring.sum() > 0:
                    arr[b] += (err[ring].sum() / pwr[ring].sum()).item()
    pigdm_b /= n
    fakgd_b /= n

    labels = [f'{edges[b]:.0f}–{edges[b+1]:.0f}' for b in range(n_bands)]
    x = np.arange(n_bands)
    w = 0.34

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6),
                                    gridspec_kw={'width_ratios': [2, 1.2], 'wspace': 0.35})

    # (a) Bar chart
    ax1.bar(x - w/2, pigdm_b, w, label=r'$\Pi$GDM', color=COLOR_PIGDM, alpha=0.85,
            edgecolor=NAVY, linewidth=0.4)
    ax1.bar(x + w/2, fakgd_b, w, label='FA-KGD', color=COLOR_FAKGD, alpha=0.85,
            edgecolor=NAVY, linewidth=0.4)
    ax1.set_xlabel('Frequency band (radius)')
    ax1.set_ylabel('NMSE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(frameon=True, fancybox=False, edgecolor=SLATE)
    ax1.grid(True, axis='y')
    _add_panel_label(ax1, '(a)')

    # (b) Relative improvement
    rel_imp = np.where(pigdm_b > 0, (pigdm_b - fakgd_b) / pigdm_b * 100, 0)
    colors_bar = [CORAL if v >= 0 else SLATE for v in rel_imp]
    ax2.barh(x, rel_imp, color=colors_bar, edgecolor=NAVY, linewidth=0.4, height=0.55)
    ax2.set_xlabel('Relative improvement (%)')
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels)
    ax2.axvline(0, color=NAVY, linewidth=0.6, linestyle='-')
    ax2.grid(True, axis='x')
    _add_panel_label(ax2, '(b)')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_frequency_analysis.pdf')
    fig.savefig(FIG_DIR / 'fig4_frequency_analysis.png')
    plt.close(fig)
    print('  ✓ Saved fig4_frequency_analysis.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 5 — Hyperparameter sensitivity (β/α heatmap)
# ═════════════════════════════════════════════════════════════

def fig5_hyperparameter_sweep():
    print('[Fig 5] Hyperparameter sweep heatmap …')
    sweep_path = OUTPUTS / 'sweep_summary.json'
    if not sweep_path.exists():
        print('  ⚠ sweep_summary.json not found, skipping.')
        return

    with open(sweep_path) as f:
        sweep = json.load(f)

    betas  = sorted(set(e['beta'] for e in sweep['grid']))
    alphas = sorted(set(e['alpha'] for e in sweep['grid']))

    delta_grid = np.zeros((len(betas), len(alphas)))
    for e in sweep['grid']:
        bi = betas.index(e['beta'])
        ai = alphas.index(e['alpha'])
        delta_grid[bi, ai] = e['delta']

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    # Navy-coral diverging colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('navy_coral',
                [SLATE, LIGHT_BLUE, WHITE, '#F8B4A8', CORAL])

    vmin, vmax = delta_grid.min(), delta_grid.max()
    im = ax.imshow(delta_grid, cmap=cmap, vmin=vmin - 0.01, vmax=vmax + 0.01,
                   aspect='auto', origin='lower')

    # Annotate cells
    for i in range(len(betas)):
        for j in range(len(alphas)):
            val = delta_grid[i, j]
            is_best = (betas[i] == sweep['best']['beta'] and
                       alphas[j] == sweep['best']['alpha'])
            weight = 'bold' if is_best else 'normal'
            color = NAVY if val > (vmin + vmax) / 2 else WHITE
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center',
                    fontsize=8, fontweight=weight, color=color)
            if is_best:
                rect = Rectangle((j-0.48, i-0.48), 0.96, 0.96,
                                 lw=2, edgecolor=NAVY, facecolor='none')
                ax.add_patch(rect)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'{a}' for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f'{b}' for b in betas])
    ax.set_xlabel(r'EMA smoothing $\alpha$')
    ax.set_ylabel(r'FPDC exponent $\beta$')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label('$\\Delta$ PSNR (dB)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_hyperparameter_sweep.pdf')
    fig.savefig(FIG_DIR / 'fig5_hyperparameter_sweep.png')
    plt.close(fig)
    print('  ✓ Saved fig5_hyperparameter_sweep.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 6 — PSNR trajectory (convergence)
# ═════════════════════════════════════════════════════════════

def fig6_psnr_trajectory():
    print('[Fig 6] PSNR trajectory …')
    run_name = 'edm_brain_R4_T50'
    if not (OUTPUTS / run_name / 'results.json').exists():
        run_name = 'edm_brain_R4'
    _, slices_data = load_results(run_name)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i, d in enumerate(slices_data):
        traj_p = d['pigdm_psnr_traj']
        traj_f = d['fakgd_psnr_traj']
        alpha = 0.25 if i > 0 else 1.0
        lw = 1.6 if i == 0 else 0.9
        ax.plot(traj_p, color=COLOR_PIGDM, alpha=alpha, lw=lw,
                label=r'$\Pi$GDM' if i == 0 else None)
        ax.plot(traj_f, color=COLOR_FAKGD, alpha=alpha, lw=lw,
                label='FA-KGD' if i == 0 else None)

    ax.set_xlabel('Diffusion step $t$')
    ax.set_ylabel('PSNR (dB)')
    ax.legend(frameon=True, fancybox=False, edgecolor=SLATE, framealpha=0.9,
              loc='lower right')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_psnr_trajectory.pdf')
    fig.savefig(FIG_DIR / 'fig6_psnr_trajectory.png')
    plt.close(fig)
    print('  ✓ Saved fig6_psnr_trajectory.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Figure 7 — Cross-domain & acceleration comparison (bar chart)
# ═════════════════════════════════════════════════════════════

def fig7_cross_domain_bar():
    print('[Fig 7] Cross-domain bar chart …')
    configs = [
        ('Brain R=4\nT=50', 'edm_brain_R4_T50', 'edm_brain_R4'),
        ('Brain R=8\nT=50', 'edm_brain_R8_T50', None),
        ('Knee R=4\nT=20',  'edm_5slices', None),
        ('Knee R=8\nT=20',  'edm_5slices_R8', None),
    ]

    labels_list, deltas, stds = [], [], []
    ssim_deltas = []
    for label, primary, fallback in configs:
        rpath = OUTPUTS / primary / 'results.json'
        if not rpath.exists() and fallback:
            rpath = OUTPUTS / fallback / 'results.json'
        if not rpath.exists():
            continue
        with open(rpath) as f:
            s = json.load(f)['summary']
        labels_list.append(label)
        deltas.append(s['delta_psnr_mean'])
        stds.append(s['delta_psnr_std'])

        # SSIM
        if 'delta_ssim_mean' in s:
            ssim_deltas.append(s['delta_ssim_mean'])
        else:
            _, sl = load_results(primary if (OUTPUTS / primary / 'results.json').exists() else fallback)
            ds = [compute_ssim(d['x_gt'], d['fakgd_recon']) - compute_ssim(d['x_gt'], d['pigdm_recon']) for d in sl]
            ssim_deltas.append(np.mean(ds))

    if not labels_list:
        print('  ⚠ No data, skipping.')
        return

    x = np.arange(len(labels_list))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={'wspace': 0.35})

    # (a) Δ PSNR bars
    bars = ax1.bar(x, deltas, yerr=stds, capsize=4, color=CORAL, alpha=0.85,
                   edgecolor=NAVY, linewidth=0.5, width=0.55, error_kw={'linewidth': 1.0})
    ax1.set_ylabel('$\\Delta$ PSNR (dB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_list, fontsize=7.5)
    ax1.axhline(0, color=NAVY, linewidth=0.5)
    ax1.grid(True, axis='y')
    _add_panel_label(ax1, '(a)')

    # Annotate values
    for bar, val in zip(bars, deltas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:+.2f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color=NAVY)

    # (b) Δ SSIM bars
    bars2 = ax2.bar(x, ssim_deltas, color=MID_BLUE, alpha=0.85,
                    edgecolor=NAVY, linewidth=0.5, width=0.55)
    ax2.set_ylabel('$\\Delta$ SSIM')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_list, fontsize=7.5)
    ax2.axhline(0, color=NAVY, linewidth=0.5)
    ax2.grid(True, axis='y')
    _add_panel_label(ax2, '(b)')

    for bar, val in zip(bars2, ssim_deltas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f'{val:+.4f}', ha='center', va='bottom', fontsize=7, fontweight='bold',
                color=NAVY)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig7_cross_domain_bar.pdf')
    fig.savefig(FIG_DIR / 'fig7_cross_domain_bar.png')
    plt.close(fig)
    print('  ✓ Saved fig7_cross_domain_bar.{pdf,png}')


# ═════════════════════════════════════════════════════════════
# Table — LaTeX results table
# ═════════════════════════════════════════════════════════════

def generate_latex_table():
    print('[Table] LaTeX results table …')
    configs = [
        ('Brain R=4, $T$=20', 'edm_brain_R4_T20', 'edm_brain_R4', 'Matched'),
        ('Brain R=4, $T$=50', 'edm_brain_R4_T50_ssim', 'edm_brain_R4_T50', 'Matched'),
        ('Brain R=4, $T$=50 (15 sl)', 'edm_brain_R4_T50_15sl', None, 'Matched'),
        ('Brain R=4, $T$=100', 'edm_brain_R4_T100', None, 'Matched'),
        ('Brain R=8, $T$=50', 'edm_brain_R8_T50', None, 'Matched'),
        ('Knee R=4, $T$=20', 'edm_5slices', None, 'Mismatched'),
        ('Knee R=8, $T$=20', 'edm_5slices_R8', None, 'Mismatched'),
    ]

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Quantitative comparison of $\Pi$GDM and FA-KGD+FPDC on fastMRI brain and knee data. '
        r'$\Delta$ denotes FA-KGD improvement over $\Pi$GDM. Best results per setting in \textbf{bold}.}',
        r'\label{tab:results}',
        r'\small',
        r'\begin{tabular}{lcccccc}',
        r'\toprule',
        r'Setting & $N$ & \multicolumn{2}{c}{PSNR (dB) $\uparrow$} & $\Delta$PSNR & \multicolumn{2}{c}{SSIM $\uparrow$} \\',
        r'\cmidrule(lr){3-4} \cmidrule(lr){6-7}',
        r' & & $\Pi$GDM & FA-KGD & (dB) & $\Pi$GDM & FA-KGD \\',
        r'\midrule',
    ]

    for label, primary, fallback, domain in configs:
        rpath = OUTPUTS / primary / 'results.json'
        if not rpath.exists() and fallback:
            rpath = OUTPUTS / fallback / 'results.json'
        if not rpath.exists():
            continue
        with open(rpath) as f:
            s = json.load(f)['summary']

        n = s['num_slices']
        pp = s['pigdm_mean_psnr']
        fp = s['fakgd_mean_psnr']
        dp = s['delta_psnr_mean']

        if 'pigdm_mean_ssim' in s:
            ps_val = s['pigdm_mean_ssim']
            fs_val = s['fakgd_mean_ssim']
        else:
            _, sl = load_results(primary if (OUTPUTS / primary / 'results.json').exists() else fallback)
            ps_list = [compute_ssim(d['x_gt'], d['pigdm_recon']) for d in sl]
            fs_list = [compute_ssim(d['x_gt'], d['fakgd_recon']) for d in sl]
            ps_val = np.mean(ps_list)
            fs_val = np.mean(fs_list)

        line = (f'{label} & {n} & {pp:.2f} & \\textbf{{{fp:.2f}}} & '
                f'{{\\color{{red}}{dp:+.2f}}} & {ps_val:.4f} & \\textbf{{{fs_val:.4f}}} \\\\')
        lines.append(line)

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    table_str = '\n'.join(lines)
    with open(FIG_DIR / 'table_results.tex', 'w') as f:
        f.write(table_str)
    print('  ✓ Saved table_results.tex')
    print(table_str)


# ═════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Generating paper figures → {FIG_DIR}/')
    print(f'Available outputs: {[d.name for d in sorted(OUTPUTS.iterdir()) if d.is_dir() and (d / "results.json").exists()]}')
    print()

    fig1_reconstruction_comparison()
    fig2_zoomed_crops()
    fig3_scaling_curve()
    fig4_frequency_analysis()
    fig5_hyperparameter_sweep()
    fig6_psnr_trajectory()
    fig7_cross_domain_bar()
    generate_latex_table()

    print(f'\n✓ All figures saved to {FIG_DIR}/')
