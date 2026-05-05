"""One-shot builder for notebooks/colab_pvas_sweep.ipynb.
Run once to (re)write the notebook. Safe to re-run; replaces cells 2..end
while preserving the first markdown header cell.
"""
import json, uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / 'notebooks' / 'colab_pvas_sweep.ipynb'

def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.splitlines(keepends=True)}

cells = []

cells.append(md("""# PVAS — Posterior-Variance-Gated Active Sampling (Colab runner)

Drives `scripts/reconstruct_mc.py` to produce **all numbers needed for the
WACV `pvas/pvas_wacv.tex` paper** in one Colab session:

1. **Main matched-budget result** at $R\\in\\{4,8\\}$, $L\\in\\{8,16,24\\}$ — five methods per cell:
   - `pigdm_mc` (static $R$ baseline)
   - `pv` (PV gate only, no acquisition)
   - `pv_active` (**ours**)
   - `random_adaptive` (mid-diffusion uniform-random — *isolates the value of PV scoring*)
   - `static_match` (matched-budget static random — *the canonical baseline*)
2. **Active-rounds ablation** $R_a\\in\\{1,2,4\\}$ at $R{=}4$, $L{=}16$.
3. **NFE-matched ablation**: ΠGDM at $T{=}60$ vs PVAS at $T{=}20$ ($\\approx 60$ NFE).
4. **Single-slice qualitative + diagnostics** (PV maps, mask evolution).
5. Summary tables + paired Wilcoxon, written into `outputs/PVAS_<RUN_TAG>/_summary.txt`.

**Hardware:** T4 (free Colab) is fine; A100 cuts the full sweep to <1 h.

**Critical:** uses `--sigma_max 10.0` at inference (matches EDM prior energy on cropped slices).
"""))

cells.append(code("""# Cell 1 — verify GPU
!nvidia-smi | head -20
"""))

cells.append(code("""# Cell 2 — clone repo (main) or git pull if already present
import os, subprocess, sys
REPO   = '/content/fastmri'
BRANCH = 'main'  # dev/multicoil-sense was merged into main
URL    = 'https://github.com/carlo-scr/fastmri.git'

if not os.path.exists(REPO):
    rc = subprocess.call(['git', 'clone', '--depth', '1', '-b', BRANCH, URL, REPO])
    if rc != 0:
        sys.exit(f'git clone failed (rc={rc}); aborting before %cd')

%cd $REPO
!git pull --ff-only origin {BRANCH} || true
!git --no-pager log -1 --oneline
"""))

cells.append(code("""# Cell 3 — install deps + preflight checks
!pip install -q h5py scikit-image scipy pyyaml
!pip install -q --upgrade "numpy>=1.26,<2.3"

import subprocess
help_mc = subprocess.run(
    ['python', 'scripts/reconstruct_mc.py', '--help'],
    check=True, capture_output=True, text=True,
).stdout
for needed in ('--active_lines', '--active_rounds', '--pv_eps_probe',
               'random_adaptive', 'pv_active', 'static_match'):
    assert needed in help_mc, (
        f'`{needed}` not found in reconstruct_mc.py --help. '
        f'Push main and re-run Cell 2.'
    )
print('OK: PV+active CLI + random_adaptive baseline present')
"""))

cells.append(code("""# Cell 4 — mount Drive, stage checkpoint + brain data
from google.colab import drive
drive.mount('/content/drive')

import os, shutil
ART = '/content/drive/MyDrive/fastmri_artifacts'
assert os.path.exists(f'{ART}/network-snapshot.pkl'), 'EDM checkpoint missing in Drive'
assert os.path.exists(f'{ART}/brain_12vols.tar'),    'brain tarball missing in Drive'

os.makedirs('checkpoints/edm/supervised_R=1', exist_ok=True)
shutil.copy(f'{ART}/network-snapshot.pkl', 'checkpoints/edm/supervised_R=1/network-snapshot.pkl')

!tar xf "$ART/brain_12vols.tar" -C /content/fastmri
!find data/multicoil_val -name '._*'    -delete
!find data/multicoil_val -name '.DS_Store' -delete

import h5py, glob
files = sorted(glob.glob('data/multicoil_val/*.h5'))
print(f'{len(files)} brain volumes ready')
"""))

cells.append(code("""# Cell 5 — smoke test (1 vol, 1 slice, R=4, T=20, all five methods)
import os
os.makedirs('outputs/_pvas_smoke', exist_ok=True)
!python scripts/reconstruct_mc.py \\
    --data_path data/multicoil_val \\
    --num_volumes 1 --num_slices 1 \\
    --acceleration 4 --center_fraction 0.08 \\
    --num_steps 20 --schedule edm --sigma_max 10.0 \\
    --target_resolution 384 320 \\
    --checkpoint_dir checkpoints/edm/supervised_R=1 \\
    --methods pigdm_mc pv pv_active random_adaptive static_match \\
    --active_lines 16 --active_rounds 2 \\
    --pv_eps_probe 0.05 \\
    --device cuda \\
    --output_dir outputs/_pvas_smoke
"""))

cells.append(md("""## §A — Main matched-budget sweep (Table 1 of the paper)

For every $(R, L)$ we run the **five methods** that share the same prior, the same sensitivity maps, and the same per-coil noise estimates. The rows of interest are:

* `pv_active` − `static_match` → headline number against the canonical baseline.
* `pv_active` − `random_adaptive` → **isolates the value of PV-guided selection** vs *any* mid-diffusion adaptation.
* `pv` − `pigdm_mc` → contribution of the gate alone.

Sweep takes ~30–60 min on T4 for the full 12-volume × 16-slice split (~192 slices), per $(R, L)$ cell.
"""))

cells.append(code("""# Cell 6 — main sweep: R x active_lines (writes outputs/PVAS_<TAG>/R{R}_a{L}/results.json)
import os
RUN_TAG = 'main_v1'
ROOT = f'outputs/PVAS_{RUN_TAG}'
os.makedirs(ROOT, exist_ok=True)

NUM_VOLS = 12
NUM_SL   = 0  # 0 = all slices in each volume

GRID = [
    (4,  8),
    (4, 16),
    (4, 24),
    (8,  8),
    (8, 16),
    (8, 24),
]

for R, AL in GRID:
    out = f'{ROOT}/R{R}_a{AL}'
    if os.path.exists(f'{out}/results.json'):
        print(f'SKIP {out} (exists)'); continue
    os.makedirs(out, exist_ok=True)
    print(f'\\n=========== R={R}  L={AL} ===========', flush=True)
    !python scripts/reconstruct_mc.py \\
        --data_path data/multicoil_val \\
        --num_volumes {NUM_VOLS} --num_slices {NUM_SL} \\
        --acceleration {R} --center_fraction 0.08 \\
        --num_steps 20 --schedule edm --sigma_max 10.0 \\
        --target_resolution 384 320 \\
        --checkpoint_dir checkpoints/edm/supervised_R=1 \\
        --methods pigdm_mc pv pv_active random_adaptive static_match \\
        --active_lines {AL} --active_rounds 2 \\
        --active_after_frac 0.5 --active_until_frac 0.85 \\
        --pv_eps_probe 0.05 \\
        --device cuda \\
        --output_dir {out} 2>&1 | tail -30
"""))

cells.append(md("""## §B — Active-rounds ablation (Table 2 of the paper)

How much does multi-round acquisition matter? Sweep $R_a \\in \\{1, 2, 4\\}$ at $R{=}4$, $L{=}16$.
Smaller workload (3 cells × 12 vols), so feel free to run this even if §A is partial.
"""))

cells.append(code("""# Cell 7 — active-rounds ablation
import os
RUN_TAG = 'rounds_v1'
ROOT = f'outputs/PVAS_{RUN_TAG}'
os.makedirs(ROOT, exist_ok=True)

R, AL = 4, 16
for Ra in (1, 2, 4):
    out = f'{ROOT}/R{R}_a{AL}_Ra{Ra}'
    if os.path.exists(f'{out}/results.json'):
        print(f'SKIP {out} (exists)'); continue
    os.makedirs(out, exist_ok=True)
    print(f'\\n=========== R_a={Ra} ===========', flush=True)
    !python scripts/reconstruct_mc.py \\
        --data_path data/multicoil_val \\
        --num_volumes 12 --num_slices 0 \\
        --acceleration {R} --center_fraction 0.08 \\
        --num_steps 20 --schedule edm --sigma_max 10.0 \\
        --target_resolution 384 320 \\
        --checkpoint_dir checkpoints/edm/supervised_R=1 \\
        --methods pv_active random_adaptive static_match \\
        --active_lines {AL} --active_rounds {Ra} \\
        --active_after_frac 0.5 --active_until_frac 0.85 \\
        --pv_eps_probe 0.05 \\
        --device cuda \\
        --output_dir {out} 2>&1 | tail -15
"""))

cells.append(md("""## §C — NFE-matched ablation (Table 3 of the paper)

PVAS uses 3× the NFE of vanilla ΠGDM (one denoiser + two centred-FD probes per step).
To isolate "policy" from "compute", compare PVAS at $T{=}20$ ($\\text{NFE}\\approx 60$)
against ΠGDM-MC at $T{=}60$ on both the static $R$ mask and the matched extended mask.
"""))

cells.append(code("""# Cell 8 — NFE-matched comparison
import os
RUN_TAG = 'nfe_v1'
ROOT = f'outputs/PVAS_{RUN_TAG}'
os.makedirs(ROOT, exist_ok=True)

R, AL = 4, 16

# (a) PVAS at T=20  (~ 3*20 = 60 NFE)
out_a = f'{ROOT}/pvas_T20'
if not os.path.exists(f'{out_a}/results.json'):
    os.makedirs(out_a, exist_ok=True)
    !python scripts/reconstruct_mc.py \\
        --data_path data/multicoil_val \\
        --num_volumes 12 --num_slices 0 \\
        --acceleration {R} --center_fraction 0.08 \\
        --num_steps 20 --schedule edm --sigma_max 10.0 \\
        --target_resolution 384 320 \\
        --checkpoint_dir checkpoints/edm/supervised_R=1 \\
        --methods pv_active \\
        --active_lines {AL} --active_rounds 2 \\
        --pv_eps_probe 0.05 \\
        --device cuda \\
        --output_dir {out_a} 2>&1 | tail -10

# (b) ΠGDM-MC at T=60 (60 NFE), static R + matched extended mask
out_b = f'{ROOT}/pigdm_T60'
if not os.path.exists(f'{out_b}/results.json'):
    os.makedirs(out_b, exist_ok=True)
    !python scripts/reconstruct_mc.py \\
        --data_path data/multicoil_val \\
        --num_volumes 12 --num_slices 0 \\
        --acceleration {R} --center_fraction 0.08 \\
        --num_steps 60 --schedule edm --sigma_max 10.0 \\
        --target_resolution 384 320 \\
        --checkpoint_dir checkpoints/edm/supervised_R=1 \\
        --methods pigdm_mc static_match \\
        --active_lines {AL} \\
        --device cuda \\
        --output_dir {out_b} 2>&1 | tail -10
"""))

cells.append(md("""## §D — Single-slice qualitative + PV-map diagnostics (Figs 1–2 of the paper)

Renders the PV map evolution and the cumulative active-mask evolution for one chosen slice
at $R{=}8$, $L{=}16$. Saves PNGs into `outputs/PVAS_diagnostics/` ready to drop into
`paper/pvas/figs/`.
"""))

cells.append(code("""# Cell 9 — diagnostic run on one slice with return_diagnostics=True
import os, sys, glob
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
sys.path.insert(0, '/content/fastmri')

from src.samplers.mri_forward import fft2c, ifft2c, create_mask
from src.samplers.schedules   import edm_sigma_schedule
from src.samplers.sense       import (
    MultiCoilSENSE, sense_combine,
    estimate_sens_maps_lowres, estimate_noise_per_coil,
)
from src.samplers.multicoil_pv import run_fakgd_mc_pv
from src.models.edm_loader     import load_edm_model, EDMDenoiser

DEV = 'cuda'
CHK = 'checkpoints/edm/supervised_R=1'
H, W = 384, 320

fp = sorted(glob.glob('data/multicoil_val/*.h5'))[0]
with h5py.File(fp, 'r') as f:
    ks = torch.from_numpy(f['kspace'][f['kspace'].shape[0] // 2])
ci = ifft2c(ks)
sh = (ci.shape[-2] - H) // 2; sw = (ci.shape[-1] - W) // 2
mc_k = fft2c(ci[..., sh:sh+H, sw:sw+W]).to(torch.complex64)
rss  = torch.sqrt((ifft2c(mc_k).abs() ** 2).sum(0))
scale = float(rss.max()); mc_k = mc_k / scale; rss = rss / scale

sig_c = estimate_noise_per_coil(mc_k).to(DEV)
mc_kn = mc_k.to(DEV)
sens  = estimate_sens_maps_lowres(mc_kn, center_fraction=0.08).to(DEV)
mask_R = create_mask(W, 0.08, 8, seed=42).expand(H, -1).to(DEV)
op    = MultiCoilSENSE(mask_R, sens, sig_c).to(DEV)

sched = edm_sigma_schedule(20, sigma_min=0.002, sigma_max=10.0).to(DEV)
net = load_edm_model(CHK, method='auto', device=DEV)
denoise = EDMDenoiser(net, device=DEV)

sigma_i_sq = torch.full((H, W), float(sig_c.mean()), device=DEV)

print('Running PVAS with diagnostics...', flush=True)
res = run_fakgd_mc_pv(
    y_mc=mask_R * mc_kn, sense_op=op,
    sigma_schedule=sched, denoiser_fn=denoise,
    sigma_i_sq_init=sigma_i_sq,
    use_pv_gate=True, refine_sens=False,
    pv_eps_probe=0.05, pv_centered=True,
    active_lines=16, active_rounds=2,
    active_score='pv', y_mc_full=mc_kn,
    seed=0, return_diagnostics=True,
)
diag = res['diagnostics']
print('PV map snapshots @ steps:', sorted(diag['P_maps'].keys()))
print('Active cols added       :', diag['active_cols'])
print('# acquired cols / step  :', diag['num_lines_per_step'])
"""))

cells.append(code("""# Cell 10 — render PV-map evolution + cumulative mask figure
import os, numpy as np, matplotlib.pyplot as plt
os.makedirs('outputs/PVAS_diagnostics', exist_ok=True)

steps = sorted(diag['P_maps'].keys())
fig, axes = plt.subplots(1, len(steps), figsize=(3 * len(steps), 3.2))
for ax, s in zip(axes, steps):
    P = np.fft.fftshift(diag['P_maps'][s])
    ax.imshow(np.log10(P + 1e-12), cmap='magma', aspect='auto')
    ax.set_title(f'step {s}'); ax.axis('off')
fig.suptitle('Posterior-variance map $\\\\log_{10} P_i(t)$ evolution (centred $k$-space)')
plt.tight_layout(); plt.savefig('outputs/PVAS_diagnostics/fig_pv_evolution.png', dpi=160)
plt.show()

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.plot(diag['num_lines_per_step'], lw=2)
ax.set_xlabel('diffusion step'); ax.set_ylabel('# acquired columns')
ax.set_title('Cumulative mask growth — R=8, L=16, R_a=2  '
             f'(final = {diag[\"num_lines_per_step\"][-1]} cols)')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('outputs/PVAS_diagnostics/fig_mask_growth.png', dpi=160)
plt.show()
"""))

cells.append(code("""# Cell 11 — qualitative recon grid (zero-fill, ΠGDM, static-match, PVAS, oracle)
import torch, numpy as np, matplotlib.pyplot as plt
from src.samplers.multicoil import run_pigdm_mc

def static_matched_mask(W, H, n_total, cf, seed, device):
    nacs = int(round(cf * W))
    if nacs % 2: nacs += 1
    nacs = min(nacs, n_total)
    n_rand = max(0, n_total - nacs)
    acs = set(range(W//2 - nacs//2, W//2 + nacs//2))
    cand = list(set(range(W)) - acs)
    rng = np.random.RandomState(seed)
    sel = rng.choice(cand, size=min(n_rand, len(cand)), replace=False) if n_rand else []
    m = np.zeros(W, dtype=np.float32)
    for c in acs: m[c] = 1
    for c in sel: m[int(c)] = 1
    return torch.from_numpy(m).expand(H, -1).to(device)

n_R = int(mask_R[0].sum().item())
mask_static = static_matched_mask(W, H, n_R + 16, 0.08, 42, DEV)
op_static = MultiCoilSENSE(mask_static, sens, sig_c).to(DEV)

zf = sense_combine(ifft2c(mask_R * mc_kn), sens).abs()
res_static = run_pigdm_mc(y_mc=mask_static * mc_kn, sense_op=op_static,
                          sigma_schedule=sched, denoiser_fn=denoise, seed=0)
res_pigR = run_pigdm_mc(y_mc=mask_R * mc_kn, sense_op=op,
                        sigma_schedule=sched, denoiser_fn=denoise, seed=0)
oracle = sense_combine(ifft2c(mc_kn), sens).abs()

fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
imgs   = [zf.cpu(), res_pigR['recon'].abs().cpu(), res_static['recon'].abs().cpu(),
          res['recon'].abs().cpu(), oracle.cpu()]
labels = ['Zero-fill', 'PIGDM-MC (static R=8)', f'Static-match ({n_R+16} cols)',
          f'PVAS ({n_R+16} cols)', 'SENSE oracle']
vmax = float(oracle.cpu().max())
for ax, im, lab in zip(axes, imgs, labels):
    ax.imshow(im, cmap='gray', vmin=0, vmax=vmax); ax.set_title(lab); ax.axis('off')
plt.tight_layout(); plt.savefig('outputs/PVAS_diagnostics/fig_recon_grid.png', dpi=160)
plt.show()
"""))

cells.append(md("""## §E — Summary tables (paper-ready)

Reads every `results.json` under `outputs/PVAS_*/` and prints
*(a)* the main matched-budget table,
*(b)* the active-rounds ablation,
*(c)* the NFE-matched comparison,
along with paired Wilcoxon tests for `pv_active` vs each baseline.
Also writes a clean text dump to `outputs/PVAS_main_v1/_summary.txt`.
"""))

cells.append(code("""# Cell 12 — main table summary + Wilcoxon
import json, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

def paired(per_slice, A, B, key='psnr_sense'):
    pairs = [(s['methods'][A][key], s['methods'][B][key])
             for s in per_slice if A in s['methods'] and B in s['methods']]
    if not pairs: return None
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    d = a - b
    try:
        _, p = wilcoxon(d, alternative='greater')
    except ValueError:
        p = float('nan')
    return d.mean(), d.std(), p, len(d)

ROOT = Path('outputs/PVAS_main_v1')
lines = []
header = (f'{\"R\":>3s} {\"L\":>3s}  {\"method\":15s}  {\"budget\":>6s}  '
          f'{\"PSNR_RSS\":>10s}  {\"PSNR_SENSE\":>11s}  {\"SSIM_SENSE\":>11s}  {\"n\":>4s}')
lines.append(header); lines.append('-' * len(header))
for R in (4, 8):
    for AL in (8, 16, 24):
        rp = ROOT / f'R{R}_a{AL}' / 'results.json'
        if not rp.exists(): continue
        d = json.load(open(rp))
        for m in ('pigdm_mc', 'pv', 'static_match', 'random_adaptive', 'pv_active'):
            v = d['summary'].get(m)
            if v is None: continue
            lines.append(f'{R:3d} {AL:3d}  {m:15s}  {v[\"budget_cols\"]:6d}  '
                         f'{v[\"psnr_rss_mean\"]:7.3f}     {v[\"psnr_sense_mean\"]:8.3f}     '
                         f'{v[\"ssim_sense_mean\"]:8.4f}     {v[\"n\"]:4d}')
        per = d.get('per_slice', [])
        for ref in ('static_match', 'random_adaptive'):
            r = paired(per, 'pv_active', ref)
            if r:
                m_, s_, p_, n_ = r
                lines.append(f'{R:3d} {AL:3d}  >> pv_active − {ref:15s}  '
                             f'Δ={m_:+.3f}±{s_:.3f} dB  Wilcoxon p={p_:.2e}  (n={n_})')
        lines.append('')

txt = '\\n'.join(lines)
print(txt)
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / '_summary.txt').write_text(txt)
print('\\nSaved →', ROOT / '_summary.txt')
"""))

cells.append(code("""# Cell 13 — ablation tables
import json
from pathlib import Path

print('=== Active-rounds ablation (R=4, L=16) ===')
print(f'{\"R_a\":>4s}  {\"PVAS\":>10s}  {\"random\":>10s}  {\"static\":>10s}  {\"Δ vs static\":>12s}  {\"Δ vs random\":>12s}')
for Ra in (1, 2, 4):
    rp = Path(f'outputs/PVAS_rounds_v1/R4_a16_Ra{Ra}/results.json')
    if not rp.exists():
        print(f'{Ra:>4d}  (missing)'); continue
    d = json.load(open(rp))['summary']
    pv  = d.get('pv_active',       {}).get('psnr_sense_mean', float('nan'))
    rnd = d.get('random_adaptive', {}).get('psnr_sense_mean', float('nan'))
    st  = d.get('static_match',    {}).get('psnr_sense_mean', float('nan'))
    print(f'{Ra:>4d}  {pv:10.3f}  {rnd:10.3f}  {st:10.3f}  {pv-st:+12.3f}  {pv-rnd:+12.3f}')

print('\\n=== NFE-matched (R=4, L=16; 60 NFE) ===')
for tag, sub, methods in [
    ('PVAS T=20',  'outputs/PVAS_nfe_v1/pvas_T20',  ('pv_active',)),
    ('PIGDM T=60', 'outputs/PVAS_nfe_v1/pigdm_T60', ('pigdm_mc', 'static_match')),
]:
    rp = Path(sub) / 'results.json'
    if not rp.exists():
        print(f'{tag}: (missing)'); continue
    s = json.load(open(rp))['summary']
    for m in methods:
        v = s.get(m)
        if v is None: continue
        print(f'  {tag:18s}  {m:14s}  PSNR_SENSE={v[\"psnr_sense_mean\"]:.3f}±{v[\"psnr_sense_std\"]:.3f}  (n={v[\"n\"]})')
"""))

cells.append(code("""# Cell 14 — back up everything to Drive
import os, tarfile
ART = '/content/drive/MyDrive/fastmri_artifacts'
out_tar = f'{ART}/pvas_results_main_v1.tar'
with tarfile.open(out_tar, 'w') as tf:
    for d in ('outputs/PVAS_main_v1',
              'outputs/PVAS_rounds_v1',
              'outputs/PVAS_nfe_v1',
              'outputs/PVAS_diagnostics'):
        if os.path.exists(d):
            tf.add(d, arcname=os.path.basename(d))
            print(f'  + {d}')
        else:
            print(f'  - {d} (missing, skipping)')
print('Saved:', out_tar)
!ls -la \"$ART\" 2>/dev/null | head -20
"""))

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f'wrote {NB_PATH}  ({len(cells)} cells)')
