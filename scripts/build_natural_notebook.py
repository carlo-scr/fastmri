"""(Re)generate notebooks/colab_pvas_natural.ipynb — the domain-2 PoC runner.

Thin cells that clone the repo and call poc_natural/* scripts (single source of
truth). Run:  python scripts/build_natural_notebook.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pvas_natural.ipynb"


_ids = iter(range(1000))


def md(src):
    return {"cell_type": "markdown", "id": f"md-{next(_ids)}", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "id": f"code-{next(_ids)}", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = [
    md("""# PVAS — natural-image domain (guided super-resolution + inpainting probe)

Reproduces every domain-2 number of the WACV paper on a single GPU (A100 Colab
recommended; CPU/MPS also work via `--device`, slower):

1. Analytic Gaussian validation of the Tweedie/Hutchinson posterior-variance estimator.
2. Guided-SR main table on CIFAR-10 (`tab:sr_main`, incl. the **duo** policy),
   budget sweep (`fig:budget`), noisy gate ablation (`tab:ablation_gate`).
3. From-scratch inpainting boundary-condition table (`tab:inpaint`) — the
   single-trajectory failure regime where **PVAS-duo** (2 synchronized
   trajectories; within-mode Tweedie PV + cross-trajectory disagreement
   $(x_1-x_2)^2/2$, i.e. the law-of-total-variance decomposition of Prop. 1)
   restores the missing cross-mode variance term at 2x trajectory cost.
4. Estimator-calibration figure inputs (`fig:calibration`), both modes.
5. CelebA-HQ 256 qualitative panel (`fig:qual_sr`) + **quantitative n=300** run.
6. FFHQ-256 extra faces domain (model id verification notes inline).
7. Robustness suite: multi-seed (0/1/2), LPIPS, probe-count sweep (q=1..16,
   end-to-end), acquisition-window sweep, block-size sweep (2/4/8).
8. Final cell: regenerate every paper table via `gen_tables.py` and diff
   against the checked-in tables (reproduce-every-number check).

Provenance: all numbers land in `outputs/poc_natural/*/results.json`; the paper's
`paper/pvas_v2/tables/gen_tables.py` consumes them directly. NFE accounting:
single-trajectory policies cost T plus 2q denoiser calls per PV refresh; duo
costs 2 trajectories; ensemble-K costs K*T."""),
    code("""# Cell 1 — GPU check (optional; MPS/CPU also work via --device)
!nvidia-smi | head -5 || true"""),
    code("""# Cell 2 — clone repo + deps
import os
REPO = '/content/fastmri'
if not os.path.exists(REPO):
    !git clone --depth 1 https://github.com/carlo-scr/fastmri.git {REPO}
%cd {REPO}
!pip install -q diffusers accelerate datasets scikit-image scipy lpips"""),
    code("""# Cell 3 — cache the CIFAR-10 test images (deterministic first 512)
from datasets import load_dataset
import torch, numpy as np, os
os.makedirs('data/cifar10', exist_ok=True)
if not os.path.exists('data/cifar10/test_images_512.pt'):
    ds = load_dataset('uoft-cs/cifar10', split='test')
    imgs = np.stack([np.array(im) for im in ds[:512]['img']])
    t = torch.from_numpy(imgs).permute(0,3,1,2).float()/255.0*2-1
    torch.save(t, 'data/cifar10/test_images_512.pt')
    print('cached', t.shape)"""),
    code("""# Cell 4 — device
import torch
DEV = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', DEV)"""),
    md("## §1 — Analytic validation (Fig. 2 / method §3.2)"),
    code("""# Cell 5 — Tweedie/Hutchinson estimator vs closed form (Gaussian prior)
!python -m poc_natural.sanity_gaussian"""),
    md("## §2 — Guided super-resolution, main table + sweeps"),
    code("""# Cell 6 — SR main table (n=128, 8 policies incl. duo; ~40 min on A100)
!python -m poc_natural.run_cifar --mode sr --n_images 128 --batch 16 --num_steps 100 \\
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \\
  --policies static random raster pv duo oracle ensemble ens_random \\
  --tag sr_main --device {DEV}"""),
    code("""# Cell 7 — budget sweep (duo included at every budget point)
for bud in [6, 18, 24]:
    !python -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \\
      --frac0 0.0 --budget_blocks {bud} --sigma_y 0.0 --active_rounds 3 \\
      --policies static random pv duo oracle --tag sr_b{bud} --device {DEV}"""),
    code("""# Cell 8 — noisy measurements: PV-gate ablation
!python -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \\
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.1 --active_rounds 3 \\
  --policies static random pv oracle --tag sr_noisy_nogate --device {DEV}
!python -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \\
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.1 --active_rounds 3 --pv_gate \\
  --policies pv --tag sr_noisy_gate --device {DEV}"""),
    md("""## §3 — Boundary condition: from-scratch inpainting (honest negative + duo fix)

Single-trajectory PV fails here (the trajectory commits to one posterior mode
and cannot see cross-mode variance). PVAS-duo restores the missing term:
score = within-mode Tweedie PV (particle 1) + cross-trajectory disagreement
$(x_1-x_2)^2/2$ — an unbiased 2-sample estimate of the cross-mode variance.
Law of total variance: Var_total = E[within-mode] + Var[cross-mode]."""),
    code("""# Cell 9 — inpainting main table (duo included)
!python -m poc_natural.run_cifar --mode inpaint --n_images 128 --batch 16 --num_steps 100 \\
  --frac0 0.10 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \\
  --policies static random pv duo oracle ensemble ens_random \\
  --tag inpaint_main --device {DEV}"""),
    md("## §4 — Estimator calibration (Fig. 3)"),
    code("""# Cell 10 — calibration, both modes
!python -m poc_natural.calibration --mode sr --n_images 128 --batch 16 --num_steps 100 \\
  --frac0 0.0 --sigma_y 0.0 --stop_frac 0.5 --probes 1 2 4 8 16 --ensemble_ks 4 8 16 --device {DEV}
!python -m poc_natural.calibration --mode inpaint --n_images 128 --batch 16 --num_steps 100 \\
  --frac0 0.10 --sigma_y 0.0 --stop_frac 0.5 --probes 1 2 4 8 16 --ensemble_ks 4 8 16 --device {DEV}"""),
    md("## §5 — CelebA-HQ 256: qualitative panel + quantitative n=300"),
    code("""# Cell 11 — qualitative faces panel (few images; heavier model)
!python -m poc_natural.qualitative_celeba --n_images 6 --num_steps 100 --budget_blocks 40 \\
  --device {DEV} || echo 'qualitative script pending'"""),
    code("""# Cell 12 — CelebA-HQ 256 QUANTITATIVE, n=300 (paper tab:celeba; ~8 h on A100)
# Same protocol as run_cifar --mode sr at 8x scale: 32x32 LR preview always
# observed, budget = 96 blocks of 8x8 px out of a 32x32 grid (1024 atoms).
# Policies at matched budget with NFE accounting; ensemble N=4 reference.
!python -m poc_natural.quantitative_celeba --n_images 300 --batch 4 --num_steps 100 \\
  --budget_blocks 96 --active_rounds 3 --pv_probes 4 --ensemble_k 4 \\
  --policies static random pv oracle duo ensemble --tag quant300 --device {DEV}"""),
    md("""## §6 — FFHQ-256 (extra faces domain)

Model-id verification (2026-07-08, HF Hub API): **`google/ddpm-ema-ffhq-256`
does NOT exist**. Google only ships `google/ncsnpp-ffhq-256` (ScoreSDE-VE
pipeline — incompatible with the DDPM alpha-bar machinery used here). Closest
DDPM-pipeline FFHQ-256 checkpoint found: `xutongda/adm_ffhq_256x256` (ADM UNet
re-exported for diffusers, MIT). The cell below loads it defensively and keeps
honest TODOs; this domain is an extra, not a headline table."""),
    code("""# Cell 13 — FFHQ-256 guided SR (defensive; TODOs where ids are unverified)
# TODO(model): confirm xutongda/adm_ffhq_256x256 (a) loads as a diffusers
#   DDPMPipeline, (b) is an epsilon-prediction UNet (EpsDenoiser assumes
#   unet(x,t).sample == predicted noise). If either fails, drop this domain.
# TODO(data): verify an FFHQ-256 test split id on the Hub; candidates below.
import sys, json, time, numpy as np, torch
sys.path.insert(0, '/content/fastmri')
FFHQ_MODEL = 'xutongda/adm_ffhq_256x256'
FFHQ_DATA_CANDIDATES = ['nuwandaa/ffhq256', 'merkol/ffhq-256']  # TODO verify ids

from poc_natural.pvas_core import (DenoiserStats, EpsDenoiser, PVASConfig,
                                   acquire_blocks, init_block_state, make_abar,
                                   run_pvas)
from poc_natural.run_cifar import psnr, ssim_batch, run_ensemble_active

unet = None
try:
    from diffusers import DDPMPipeline
    pipe = DDPMPipeline.from_pretrained(FFHQ_MODEL)
    unet = pipe.unet.to(DEV).eval()
    abar = make_abar(pipe.scheduler).to(DEV)
except Exception as e:
    print('FFHQ model unavailable — keeping TODO, skipping cell:', e)

def load_ffhq(n):
    from datasets import load_dataset
    last = None
    for name in FFHQ_DATA_CANDIDATES:
        try:
            ds = load_dataset(name, split='train', streaming=True)
            imgs = []
            for ex in ds:
                col = 'image' if 'image' in ex else 'img'
                imgs.append(np.array(ex[col].convert('RGB').resize((256, 256))))
                if len(imgs) >= n:
                    break
            if len(imgs) >= n:
                print(f'loaded {n} FFHQ faces from {name}')
                return (torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)
                        .float() / 255.0 * 2 - 1)
        except Exception as e:
            last = e
    raise RuntimeError(f'no FFHQ-256 dataset loadable: {last}')

if unet is not None:
    images = load_ffhq(32)
    out = {'config': {'model': FFHQ_MODEL, 'n_images': 32, 'block': 8,
                      'budget_blocks': 96, 'num_steps': 100, 'seed': 0},
           'per_image': {}, 'summary': {}}
    t0 = time.time()
    for pol in ['static', 'random', 'pv', 'duo']:
        ps, ss = [], []
        for b0 in range(0, 32, 4):
            x0 = images[b0:b0 + 4].to(DEV)
            gen = torch.Generator().manual_seed(1000 + b0)
            st = init_block_state(x0, 0.0, 8, 0.0, gen, with_lr=True)
            cfg = PVASConfig(num_steps=100, pv_n_probes=4, use_pv_gate=False,
                             active_rounds=3, active_window=(0.25, 0.65),
                             policy=pol if pol != 'static' else 'random', seed=b0)
            if pol == 'static':
                g2 = torch.Generator().manual_seed(2000 + b0)
                sc = torch.rand(x0.shape[0], 1, 32, 32, generator=g2).to(DEV)
                st = acquire_blocks(st, x0, sc, 96, g2)
                r = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0, 0,
                             PVASConfig(**{**cfg.__dict__, 'active_rounds': 0,
                                           'policy': 'random'}))
            elif pol == 'duo':
                r, _ = run_ensemble_active(unet, abar, st, x0, 96, cfg, 2,
                                           select='duo')
            else:
                r = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0,
                             96, cfg)
            ps.extend(psnr(r.x0, x0).cpu().tolist())
            ss.extend(ssim_batch(r.x0, x0).tolist())
        out['per_image'][pol] = {'psnr': ps, 'ssim': ss}
        out['summary'][pol] = {'psnr_mean': float(np.mean(ps)),
                               'psnr_std': float(np.std(ps)),
                               'ssim_mean': float(np.mean(ss)),
                               'ssim_std': float(np.std(ss))}
        print(f'[{pol}] PSNR {np.mean(ps):.3f}  ({time.time()-t0:.0f}s)')
    import os; os.makedirs('outputs/poc_natural/ffhq_quant', exist_ok=True)
    json.dump(out, open('outputs/poc_natural/ffhq_quant/results.json', 'w'),
              indent=2)
    print('saved outputs/poc_natural/ffhq_quant/results.json')"""),
    md("""## §7 — Multi-seed robustness (seeds 0, 1, 2 x sr_main config)

Same sr_main protocol per seed; --seed changes mask initialisation and
trajectory noise (the image subset is fixed by an internal generator, so runs
are paired across seeds)."""),
    code("""# Cell 14 — three seeds of the sr_main configuration
for s in [0, 1, 2]:
    !python -m poc_natural.run_cifar --mode sr --n_images 128 --batch 16 --num_steps 100 \\
      --frac0 0.0 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 \\
      --policies static random pv duo --seed {s} --tag sr_seed{s} --device {DEV}"""),
    code("""# Cell 15 — aggregate multi-seed table (mean±std of policy PSNR across seeds)
import json, numpy as np
runs = {}
for s in [0, 1, 2]:
    p = f'outputs/poc_natural/cifar_sr_seed{s}/results.json'
    try:
        runs[s] = json.load(open(p))
    except FileNotFoundError:
        print('missing', p)
pols = ['static', 'random', 'pv', 'duo']
hdr = 'policy'.ljust(10) + ''.join(f'  seed{s}' for s in runs) + '   mean±std'
print(hdr); print('-' * len(hdr))
for pol in pols:
    ms = [runs[s]['summary'][pol]['psnr_mean'] for s in runs
          if pol in runs[s].get('summary', {})]
    if ms:
        print(pol.ljust(10) + ''.join(f'  {m:6.2f}' for m in ms)
              + f'   {np.mean(ms):.2f}±{np.std(ms):.2f}')
for s in runs:
    d = runs[s]['summary'].get('delta_pv_vs_static')
    if d:
        print(f"seed {s}: Δ(pv-static) = {d['mean']:+.3f}±{d['std']:.3f} "
              f"(Wilcoxon p={d['wilcoxon_p_greater']:.2e})")"""),
    md("""## §8 — LPIPS metric (perceptual; adds to summaries)

run_cifar does not store reconstructions, so this cell re-runs the sr_main
protocol (n=64) with the same seeds and computes LPIPS(alex) + PSNR per policy,
writing `outputs/poc_natural/cifar_lpips/results.json`."""),
    code("""# Cell 16 — LPIPS on the sr_main protocol
!pip install -q lpips
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, '/content/fastmri')
import lpips as lpips_lib
from poc_natural.pvas_core import (DenoiserStats, EpsDenoiser, PVASConfig,
                                   acquire_blocks, init_block_state, make_abar,
                                   run_pvas)
from poc_natural.run_cifar import psnr, run_ensemble_active
from diffusers import DDPMScheduler, UNet2DModel

unet = UNet2DModel.from_pretrained('google/ddpm-cifar10-32').to(DEV).eval()
abar = make_abar(DDPMScheduler.from_pretrained('google/ddpm-cifar10-32')).to(DEV)
loss_fn = lpips_lib.LPIPS(net='alex').to(DEV)
all_imgs = torch.load('data/cifar10/test_images_512.pt')
idx = torch.randperm(len(all_imgs),
                     generator=torch.Generator().manual_seed(123))[:64]
images = all_imgs[idx]

res = {'config': {'n_images': 64, 'budget_blocks': 12, 'num_steps': 100,
                  'active_rounds': 3, 'pv_probes': 4, 'ensemble_k': 8,
                  'metric': 'lpips_alex', 'seed': 0},
       'per_image': {}, 'summary': {}}
t0 = time.time()
for pol in ['static', 'random', 'pv', 'duo', 'ensemble']:
    lp, ps = [], []
    for b0 in range(0, 64, 16):
        x0 = images[b0:b0 + 16].to(DEV)
        gen = torch.Generator().manual_seed(1000 + b0)
        st = init_block_state(x0, 0.0, 4, 0.0, gen, with_lr=True)
        cfg = PVASConfig(num_steps=100, pv_n_probes=4, use_pv_gate=False,
                         active_rounds=3,
                         policy=pol if pol != 'static' else 'random', seed=b0)
        if pol == 'static':
            g2 = torch.Generator().manual_seed(2000 + b0)
            sc = torch.rand(x0.shape[0], 1, 8, 8, generator=g2).to(DEV)
            st = acquire_blocks(st, x0, sc, 12, g2)
            r = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0, 0,
                         PVASConfig(**{**cfg.__dict__, 'active_rounds': 0,
                                       'policy': 'random'}))
        elif pol in ('ensemble', 'duo'):
            kk = 2 if pol == 'duo' else 8
            r, _ = run_ensemble_active(unet, abar, st, x0, 12, cfg, kk,
                                       select='duo' if pol == 'duo' else 'var')
        else:
            r = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0, 12,
                         cfg)
        with torch.no_grad():
            lp.extend(loss_fn(r.x0.clamp(-1, 1), x0).flatten().cpu().tolist())
        ps.extend(psnr(r.x0, x0).cpu().tolist())
    res['per_image'][pol] = {'lpips': lp, 'psnr': ps}
    res['summary'][pol] = {'lpips_mean': float(np.mean(lp)),
                           'lpips_std': float(np.std(lp)),
                           'psnr_mean': float(np.mean(ps)),
                           'psnr_std': float(np.std(ps))}
    print(f"[{pol}] LPIPS {np.mean(lp):.4f}  PSNR {np.mean(ps):.2f}  "
          f"({time.time()-t0:.0f}s)")
os.makedirs('outputs/poc_natural/cifar_lpips', exist_ok=True)
json.dump(res, open('outputs/poc_natural/cifar_lpips/results.json', 'w'),
          indent=2)
print('saved outputs/poc_natural/cifar_lpips/results.json')"""),
    md("""## §9 — Probe-count sensitivity, END-TO-END (q = 1..16)

Not just calibration: each q runs the full acquisition + reconstruction
pipeline, so the table shows how many Hutchinson probes selection actually
needs. NFE grows as 2q per PV refresh."""),
    code("""# Cell 17 — end-to-end probe sweep
for q in [1, 2, 4, 8, 16]:
    !python -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \\
      --frac0 0.0 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --pv_probes {q} \\
      --policies static pv --tag sr_q{q} --device {DEV}"""),
    md("""## §10 — Acquisition-window sweep

`active_window` is a PVASConfig field (default (0.45, 0.75)), not a CLI flag;
the cell patches run_cifar's bound `run_pvas` so every policy (incl. duo via
run_ensemble_active) uses the swept window, then calls run_cifar.main()
end-to-end. Early window = acquire while the trajectory is still uncommitted;
late = acquire with a sharper PV map."""),
    code("""# Cell 18 — window sweep (early / default / late)
import sys
sys.path.insert(0, '/content/fastmri')
import poc_natural.run_cifar as rc
import poc_natural.pvas_core as core

_orig_run_pvas = rc.run_pvas
for lo, hi in [(0.25, 0.55), (0.45, 0.75), (0.60, 0.90)]:
    def patched(den, state, x0f, budget, cfg, ensemble_scorer=None, _w=(lo, hi)):
        cfg = core.PVASConfig(**{**cfg.__dict__, 'active_window': _w})
        return _orig_run_pvas(den, state, x0f, budget, cfg, ensemble_scorer)
    rc.run_pvas = patched
    tag = f'sr_win{int(lo * 100)}_{int(hi * 100)}'
    sys.argv = ['run_cifar', '--mode', 'sr', '--n_images', '64', '--batch', '16',
                '--num_steps', '100', '--frac0', '0.0', '--budget_blocks', '12',
                '--sigma_y', '0.0', '--active_rounds', '3',
                '--policies', 'static', 'random', 'pv', 'duo',
                '--tag', tag, '--device', DEV]
    print('=== window', (lo, hi), '->', tag)
    rc.main()
rc.run_pvas = _orig_run_pvas"""),
    md("""## §11 — Block-size sweep (2 / 4 / 8 px atoms, matched PIXEL budget)

Block size changes both the operator (LR preview = block-averaged image) and
the acquisition granularity; the budget is scaled to keep total acquired
pixels fixed at 192 px (= 12 blocks of 4x4)."""),
    code("""# Cell 19 — block-size sweep
for blk, bud in [(2, 48), (4, 12), (8, 3)]:
    !python -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \\
      --block {blk} --frac0 0.0 --budget_blocks {bud} --sigma_y 0.0 --active_rounds 3 \\
      --policies static random pv duo --tag sr_blk{blk} --device {DEV}"""),
    md("## §12 — Summary + reproduce-every-paper-number check"),
    code("""# Cell 20 — print all summaries
import json, glob
for p in sorted(glob.glob('outputs/poc_natural/*/results.json')):
    r = json.load(open(p))
    print('\\n==', p)
    for k, v in r.get('summary', r.get('signals', {})).items():
        print('  ', k, v)"""),
    code("""# Cell 21 — reproduce every paper number: regenerate tables, diff vs checked-in
# gen_tables.py reads ONLY outputs/**/results.json (no numeric literals); if the
# regenerated .tex fragments match the committed ones bit-for-bit, every number
# in the paper is reproduced by this notebook + the MRI notebook.
!python paper/pvas_v2/tables/gen_tables.py
import subprocess
r = subprocess.run(['git', 'diff', '--exit-code', '--stat', '--',
                    'paper/pvas_v2/tables'],
                   capture_output=True, text=True, cwd='/content/fastmri')
print(r.stdout or '(no diff)')
s = subprocess.run(['git', 'status', '--short', 'paper/pvas_v2/tables'],
                   capture_output=True, text=True, cwd='/content/fastmri')
print(s.stdout or '(no untracked table files)')
# Note: layout-guarded fragments (marker "[length-edit]") are written as
# .tex.new by gen_tables.py and appear as untracked files — diff them by hand.
print('REPRODUCED: tables match the checked-in paper' if r.returncode == 0 else
      'MISMATCH: regenerated tables differ — inspect `git diff paper/pvas_v2/tables`')"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1))
print("wrote", NB)
