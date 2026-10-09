"""Generate pilot_colab.ipynb (run: python build_notebook.py).

Run a *copy* (pilot_colab_run.ipynb): rebuilding overwrites this file's outputs.
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md(r"""
# MIDL 2027 pilot: repetitions as supervision for low-field diffusion priors

Plan and gates: `MIDL2027/proposal_repetitions/proposal.md`. Every stage writes resumable output
to Drive (`fastmri_artifacts/midl27_reps_results`). M4Raw files are **range-fetched** from Zenodo
(the zips are stored uncompressed), so nothing large is downloaded.

| Stage | Question | Gate | A100 time (est.) |
|---|---|---|---|
| P0 diagnostics (10 val subjects) | is the data usable? | kill if steps below floor > 0.30, physiological fraction > 0.5, estimator check outside [0.9, 1.2] | ~5 min (CPU) |
| P1 training set (40 train subjects) | — | — | ~20 min (fetch + ESPIRiT) |
| P2 post-training + unbiased denoising test | does the denoiser improve? | ≥ 10% lower N2N MSE at the floor | ~30 min |
| P3 reconstruction (tune 4 / eval 12 val subjects) | is there a paper? | **GO**: ≥ +1 dB over zero-shot (CI > 0) at R = 2 and 4, and beats the fully sampled **2-average** at R = 2 in ≥ 50% of units | ~1.5 h |

Runtime: A100 GPU. Before running: this branch pushed to GitHub, `network-snapshot.pkl` (3 T brain
EDM) in `MyDrive/fastmri_artifacts/`.
""")

code(r"""
# 0.1  config
BRANCH = 'research/sketchsense-wacv27'
ADPS_COMMIT = '75feb2fad4b53e48791fcb0257f2e53a25340774'
SEED = 2027
N_P0, N_TUNE, N_EVAL = 10, 4, 12          # val subjects (disjoint tune/eval; P0 may overlap)
N_TRAIN, TRAIN_SLICES = 40, 8              # train subjects, central slices per volume
STEPS, BATCH, LR = 6000, 8, 2e-5           # post-training
T, RS, KAPPAS, LAMDAS = 50, (2, 4), (0.3, 1.0, 3.0), (0.03, 0.1)
EVAL_SLICES = 4
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
# 0.2  code + dependencies
import os
%cd /content
if not os.path.isdir('/content/fastmri/.git'):
    !git clone -q https://github.com/carlo-scr/fastmri.git
%cd /content/fastmri
!git fetch -q origin && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH
if not os.path.isdir('external/adps/dnnlib'):
    !git clone -q https://github.com/utcsilab/ambient-diffusion-mri.git external/adps
!cd external/adps && git checkout -q $ADPS_COMMIT
!git --no-pager log -1 --oneline
!pip install -q sigpy s3fs
""")

code(r"""
# 0.3  storage, model, subject splits
import sys, json, copy, time
sys.path.insert(0, '/content/fastmri'); sys.path.insert(0, '/content/fastmri/experiments/midl27_reps')
import numpy as np, torch, pandas as pd
import m4raw as M, pilot as P, train_reps as TR, storage
from src.models.edm_loader import load_edm_model, EDMDenoiser
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

D = storage.Drive()
OUT, RAW = D.out, '/content/m4raw_raw'
CK = 'checkpoints/edm/supervised_R=1'
D.get_input('network-snapshot.pkl', f'{CK}/network-snapshot.pkl')
base_net = load_edm_model(CK, method='auto', device='cuda')

lst = {s: M.zip_listing(s, cache=f'{OUT}/listing_{s}.json') for s in ('train', 'val')}
cat = {s: M.catalog(lst[s]) for s in lst}
SPLITS = f'{OUT}/splits.json'                       # frozen on first run
if not os.path.exists(SPLITS):
    val = M.pick_subjects(cat['val'], None, SEED)
    rng = np.random.default_rng(SEED); perm = list(rng.permutation(val))
    json.dump(dict(tune=sorted(perm[:N_TUNE]), eval=sorted(perm[N_TUNE:N_TUNE + N_EVAL]),
                   p0=sorted(perm[:N_P0]), train=M.pick_subjects(cat['train'], N_TRAIN, SEED),
                   created=time.strftime('%Y-%m-%d %H:%M:%S')), open(SPLITS, 'w'), indent=1)
    D.sync()
SP = json.load(open(SPLITS))
print({k: (len(v) if isinstance(v, list) else v) for k, v in SP.items()})
""")

md(r"""
## P0 — data diagnostics and kill criteria
""")

code(r"""
P0 = f'{OUT}/p0_diagnostics.json'
if not os.path.exists(P0):
    rows = P.diagnostics('val', SP['p0'], cat['val'], lst['val'], RAW, n_slices=4, T=T)
    json.dump(rows, open(P0, 'w'), indent=1); D.sync()
p0 = pd.DataFrame(json.load(open(P0)))
summ = p0.groupby('contrast')[['snr_median', 'floor', 'frac_steps_below_floor', 'physio_frac', 'ena_full1',
                               'check_full_2avg_ena', 'max_shift_px', 'max_phase_deg', 'psi_max_corr']].median()
display(summ.round(3))
kill = dict(steps_below_floor=float(p0.frac_steps_below_floor.median()) > 0.30,
            physio_dominates=float(p0.physio_frac.median()) > 0.5,
            estimator_off=not (0.9 <= float(p0.ena_full1.median()) <= 1.2))
print('kill criteria:', kill, '->', 'STOP' if any(kill.values()) else 'continue')
""")

md(r"""
## P1 — training set (repetition pairs, noise made white)
""")

code(r"""
TRAIN_NPZ = f'{OUT}/p1_train_{N_TRAIN}subj_{TRAIN_SLICES}sl.npz'
train_slices = P.build_training('train', SP['train'], cat['train'], lst['train'], RAW,
                                n_slices=TRAIN_SLICES, cache=TRAIN_NPZ)
D.sync()
heldout = P.build_training('val', SP['tune'], cat['val'], lst['val'], RAW, n_slices=4,
                           cache=f'{OUT}/p1_heldout_tune.npz')
print(len(train_slices), 'training slices;', len(heldout), 'held-out slices;',
      'median floor', np.median([s['floor'] for s in train_slices]).round(4))
""")

md(r"""
## P2 — post-training and the unbiased denoising test
Differences in Noise2Noise MSE between two denoisers on the same held-out pairs are unbiased
differences in true denoising error (the target noise cancels).
""")

code(r"""
data = TR.RepPairs(train_slices, torch.device('cuda'))
ft_net, hist = TR.finetune(copy.deepcopy(base_net), data, steps=STEPS, batch=BATCH, lr=LR,
                           out_dir='/content/ft_ckpt', ckpt_every=1000, log_every=200, sync=None)
torch.save(ft_net.state_dict(), f'{OUT}/p2_ft_ema.pt'); json.dump(hist, open(f'{OUT}/p2_hist.json', 'w')); D.sync()
""")

code(r"""
ho = TR.RepPairs(heldout, torch.device('cuda'))
floor = float(np.median([s['floor'] for s in heldout]))
dn = pd.DataFrame(TR.denoising_eval({'zeroshot': base_net.eval(), 'ft': ft_net.eval()}, ho,
                                    sigmas=[floor, 0.1, 0.2, 0.5, 1.0], n_pairs=256))
dn['ft_vs_zeroshot_%'] = 100 * (dn.ft - dn.zeroshot) / dn.zeroshot
display(dn.round(5)); dn.to_csv(f'{OUT}/p2_denoising.csv', index=False); D.sync()
print('P2 gate (>= 10% lower at the floor):', 'continue' if dn['ft_vs_zeroshot_%'].iloc[0] <= -10 else 'WEAK')
""")

md(r"""
## P3 — reconstruction: tune on 4 val subjects, evaluate on 12
""")

code(r"""
dz, df = EDMDenoiser(base_net.eval(), 'cuda'), EDMDenoiser(ft_net.eval(), 'cuda')
METHODS = {'zf': dict(kind='zf'), 'full1': dict(kind='full1'), 'l1': dict(kind='l1'),
           'zeroshot': dict(kind='sampler', denoiser=dz),
           'ft_stop': dict(kind='sampler', denoiser=df, stop_at_floor=True),
           'ft_two': dict(kind='sampler', denoiser=df, two_regime=True, below=dz)}
TUNE = f'{OUT}/p3_tune.jsonl'
P.evaluate('val', SP['tune'], cat['val'], lst['val'], RAW, TUNE, methods=METHODS, Rs=RS, T=T,
           kappas=KAPPAS, lamdas=LAMDAS, n_slices=EVAL_SLICES, sync=D.sync)
BEST = P.select_params(TUNE); json.dump(BEST, open(f'{OUT}/p3_best.json', 'w'), indent=1); D.sync()
print(json.dumps(BEST, indent=1))
""")

code(r"""
EVAL = f'{OUT}/p3_eval.jsonl'
for R in RS:
    for name, m in METHODS.items():
        k = BEST.get(f'{name}|R{R}')
        P.evaluate('val', SP['eval'], cat['val'], lst['val'], RAW, EVAL, methods={name: m}, Rs=(R,), T=T,
                   kappas=(k,) if m['kind'] == 'sampler' else KAPPAS, lamdas=(k,) if m['kind'] == 'l1' else LAMDAS,
                   n_slices=EVAL_SLICES, sync=D.sync)
G = P.gate_report(EVAL)
json.dump(G, open(f'{OUT}/p3_gate.json', 'w'), indent=1, default=float); D.sync()
for R, rows in G['per_R'].items():
    print(f'\nR = {R}'); display(pd.DataFrame(rows).T.round(3))
print('\nbest post-trained variant:', G['best_ft'], '| GO:', G['GO'], '| STRONG:', G['STRONG'])
""")

code(r"""
# per-contrast view (T1/T2 have 3 repetitions -> common-reference comparisons; FLAIR has 2)
ev = P.load_records(EVAL)
display(ev.groupby(['R', 'contrast', 'method'])[['psnr_corr', 'ena']].median().unstack('method').round(2))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU",
                                   "colab": {"provenance": [], "gpuType": "A100"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("pilot_colab.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
