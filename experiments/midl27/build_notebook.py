"""Generate experiments/midl27/midl27_colab.ipynb (run: python build_notebook.py)."""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# MIDL 2027 — frequency-resolved data consistency (FA-KGD): experiment book

Fills the `XX.XX` cells of `MIDL2027/paper`. Every stage writes **resumable JSONL** straight
to Google Drive, so a disconnected runtime loses at most one reconstruction: just re-run the cell.

| Stage | Decides | Paper output | A100 time (est.) |
|---|---|---|---|
| 0 smoke | pipeline works, per-step cost | — | 5 min |
| **1 = Gate G1 (by Oct 12)** | is there a main-track paper? | go / no-go | ~2.5 h |
| 2 = Gate G2 (by Nov 1) | plug-in + step-robustness claims | Tab. 3, Tab. 4, Fig. 1, Fig. 2 | ~8–10 h |
| 3 = Gate G3 (by Nov 8) | low-SNR claim | Tab. 5 | ~2 h |
| 4 ablations + compute | what stays in the method | Tab. 7, Tab. 8 | ~3 h |

Estimates assume ~25 ms per EDM evaluation at 384×320 on an A100; **stage 0 measures the real
number**, re-plan if it is >2× off.

**Before running:** (1) push this branch to GitHub; (2) put `network-snapshot.pkl` (supervised
EDM, R=1) and `midl27_brain.tar` (from `experiments/midl27/pack_data.py`) in
`MyDrive/fastmri_artifacts/`; (3) Runtime → A100 GPU.

Protocol (paper Sec. 4): calibration volumes are used only to tune; test volumes are frozen in the
manifest before any test run; isotropic baselines are tuned **per (R, T)**; FA-KGD gets **one**
configuration per host chosen at T = 50. The unit of analysis is the volume.
""")

code(r"""
# 0.1  GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code(r"""
# 0.2  code: this repo (your branch) + the ADPS source the EDM pickle needs (pinned commit)
import os
BRANCH = 'research/sketchsense-wacv27'     # <- the branch you pushed
ADPS_COMMIT = '75feb2fad4b53e48791fcb0257f2e53a25340774'
%cd /content
if not os.path.isdir('/content/fastmri/.git'):
    !git clone -q https://github.com/carlo-scr/fastmri.git
%cd /content/fastmri
!git fetch -q origin && git checkout -q $BRANCH && git reset -q --hard origin/$BRANCH
if not os.path.isdir('external/adps/dnnlib'):
    !git clone -q https://github.com/utcsilab/ambient-diffusion-mri.git external/adps
!cd external/adps && git checkout -q $ADPS_COMMIT
!git --no-pager log -1 --oneline
""")

code(r"""
# 0.3  dependencies (Colab already has torch, scipy, scikit-image, pandas, h5py)
!pip install -q sigpy s3fs pyyaml
""")

code(r"""
# 0.4  Drive: checkpoint + data in, results out.
# Tries drive.mount; if that fails (it does in the VS Code Colab extension, and intermittently
# in the browser: colabtools#5944), falls back to the Drive API. In API mode OUT is a local
# folder restored from Drive now and synced back every 5 min and after every go() call.
import os, io, time, shutil, subprocess, threading
DRIVE_DIR = 'fastmri_artifacts'                 # folder in My Drive
DATA = '/content/data/midl27_cohort'
TAR_NAME, TAR_BYTES = 'midl27_brain.tar', 4584202240   # size of data/midl27_brain.tar
PKL_NAME = 'network-snapshot.pkl'

try:
    from google.colab import drive
    drive.mount('/content/drive')
    MODE = 'mount'
except Exception as e:
    print(f'drive.mount failed ({type(e).__name__}: {str(e)[:120]}) -> using the Drive API')
    MODE = 'api'

if MODE == 'mount':
    ART = f'/content/drive/MyDrive/{DRIVE_DIR}'
    OUT = f'{ART}/midl27_results'
    def fetch_input(name, dst): shutil.copy(f'{ART}/{name}', dst)
    def input_size(name): return os.path.getsize(f'{ART}/{name}') if os.path.exists(f'{ART}/{name}') else None
    def sync(verbose=False): pass               # OUT already lives on Drive
else:
    from google.colab import auth
    auth.authenticate_user()
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
    _svc = build('drive', 'v3', cache_discovery=False)
    _lock = threading.Lock()

    def _find(name, parent='root', folder=False):
        q = f"name = '{name}' and '{parent}' in parents and trashed = false"
        if folder: q += " and mimeType = 'application/vnd.google-apps.folder'"
        r = _svc.files().list(q=q, fields='files(id,name,size,modifiedTime)', pageSize=10).execute()['files']
        return r[0] if r else None
    def _folder(name, parent='root'):
        f = _find(name, parent, folder=True)
        if f: return f['id']
        meta = {'name': name, 'parents': [parent], 'mimeType': 'application/vnd.google-apps.folder'}
        return _svc.files().create(body=meta, fields='id').execute()['id']

    ART_ID = _find(DRIVE_DIR, folder=True)
    assert ART_ID, f'no folder "{DRIVE_DIR}" in My Drive'
    ART_ID = ART_ID['id']; OUT_ID = _folder('midl27_results', ART_ID)
    ART, OUT = f'drive-api:{DRIVE_DIR}', '/content/midl27_results'

    def input_size(name):
        f = _find(name, ART_ID); return int(f['size']) if f else None
    def _download(file_id, dst, chunk=2**26):
        with io.FileIO(dst + '.part', 'wb') as fh:
            dl = MediaIoBaseDownload(fh, _svc.files().get_media(fileId=file_id), chunksize=chunk)
            done, t0 = False, time.time()
            while not done:
                st, done = dl.next_chunk(num_retries=5)
                if st: print(f'\r  {os.path.basename(dst)}: {st.progress():.0%} ({time.time()-t0:.0f}s)', end='')
        os.replace(dst + '.part', dst); print()
    def fetch_input(name, dst):
        f = _find(name, ART_ID); assert f, f'{name} not in {DRIVE_DIR}'
        _download(f['id'], dst)

    _synced = {}                                 # local path -> (size, mtime) last uploaded
    def sync(verbose=False):
        with _lock:
            n = 0
            for fn in sorted(os.listdir(OUT)):
                p = f'{OUT}/{fn}'
                if not os.path.isfile(p) or fn.endswith('.part'): continue
                st = (os.path.getsize(p), os.path.getmtime(p))
                if _synced.get(p) == st: continue
                media = MediaFileUpload(p, resumable=st[0] > 2**22)
                f = _find(fn, OUT_ID)
                if f: _svc.files().update(fileId=f['id'], media_body=media).execute(num_retries=5)
                else: _svc.files().create(body={'name': fn, 'parents': [OUT_ID]}, media_body=media,
                                          fields='id').execute(num_retries=5)
                _synced[p] = st; n += 1
            if verbose or n: print(f'[sync] {n} file(s) -> Drive/{DRIVE_DIR}/midl27_results')

    # restore earlier results so resumable runs continue where they stopped
    os.makedirs(OUT, exist_ok=True)
    r = _svc.files().list(q=f"'{OUT_ID}' in parents and trashed = false",
                          fields='files(id,name)', pageSize=1000).execute()['files']
    for f in r:
        dst = f'{OUT}/{f["name"]}'
        if not os.path.exists(dst):
            _download(f['id'], dst)
        _synced[dst] = (os.path.getsize(dst), os.path.getmtime(dst))
    print(f'restored {len(r)} result file(s) from Drive')

    def _bg():
        while True:
            time.sleep(300)
            try: sync()
            except Exception as e: print('[sync] failed, will retry:', e)
    threading.Thread(target=_bg, daemon=True).start()

os.makedirs(OUT, exist_ok=True)
os.makedirs('checkpoints/edm/supervised_R=1', exist_ok=True)
if not os.path.exists('checkpoints/edm/supervised_R=1/network-snapshot.pkl'):
    fetch_input(PKL_NAME, 'checkpoints/edm/supervised_R=1/network-snapshot.pkl')
if not os.path.isdir(DATA):
    size = input_size(TAR_NAME)
    if size is None:
        raise FileNotFoundError(f'{TAR_NAME} not found in {DRIVE_DIR}')
    if size != TAR_BYTES:
        raise IOError(f'{TAR_NAME} is {size} bytes, expected {TAR_BYTES}: '
                      'upload incomplete or still syncing; re-upload and retry')
    # copy to local disk, extract into a temp dir and rename: no partial cohort on failure
    os.makedirs('/content/data', exist_ok=True)
    local_tar = f'/content/data/{TAR_NAME}'
    if not os.path.exists(local_tar) or os.path.getsize(local_tar) != TAR_BYTES:
        fetch_input(TAR_NAME, local_tar)
    tmp = '/content/data/_extract'
    shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    r = subprocess.run(['tar', 'xf', local_tar, '-C', tmp], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f'tar failed ({r.returncode}): {r.stderr.strip()[-2000:]}')
    os.replace(f'{tmp}/midl27_cohort', DATA); shutil.rmtree(tmp); os.remove(local_tar)
n_vol = len([f for f in os.listdir(DATA) if f.endswith('.h5')])
print(f'MODE={MODE}  OUT={OUT}  {n_vol} volumes in {DATA}')
assert n_vol == 28, 'partial cohort: !rm -rf /content/data/midl27_cohort and re-run this cell'
""")

code(r"""
# 0.5  imports and a tiny helper
import sys, json, time, copy
sys.path.insert(0, '/content/fastmri'); sys.path.insert(0, '/content/fastmri/experiments/midl27')
import torch, pandas as pd
import run, analysis, fdc
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE = dict(manifest=f'{OUT}/manifest.json', checkpoint='checkpoints/edm/supervised_R=1',
            device='cuda', sigma_max=10.0, mask='equispaced', seeds=[0])

def go(**kw):
    cfg = {**BASE, **kw}
    try:
        return run.run_grid(cfg)
    finally:
        sync()            # no-op when Drive is mounted

def vol(path):
    return analysis.volume_table(analysis.load(path))
""")

code(r"""
# 0.6  frozen manifest: 4 calibration + 24 test volumes (never overwritten once written)
if not os.path.exists(BASE['manifest']):
    man = run.make_manifest(DATA, BASE['manifest'], n_calib=4, n_test=24, acquisition='AXFLAIR')
man = json.load(open(BASE['manifest']))
sync()
print('calib:', len(man['calib']), ' test:', len(man['test']), ' created', man['created'])
""")

md(r"""
## Stage 0 — smoke test and cost measurement
One slice, every method, T = 10. Checks that nothing diverges and measures seconds per step.
""")

code(r"""
SMOKE = f'{OUT}/smoke.jsonl'
std = list(fdc.standard_methods())
go(split='test', n_volumes=1, n_slices=1, R=[8], T=[10], methods=std, output=SMOKE)
s = analysis.load(SMOKE)
s['s_per_step'] = s.runtime_s / s['T']
display(s[['method', 'psnr', 'ssim', 'nfe', 'runtime_s', 's_per_step']].round(3))
""")

md(r"""
## Stage 1 — Gate G1 (decide by **12 Oct 2026**)
G1 passes if, on **real (prewhitened) noise**, +FA-KGD beats the **tuned** isotropic baseline by
≥ 0.2 dB (volume-level mean, 95 % bootstrap CI excluding 0) at T ≥ 50 (revision plan C1–C3).

### 1a — C3: is the CS231N M-step inert?
""")

code(r"""
MSTEP = f'{OUT}/g1_mstep.jsonl'
go(split='calib', n_slices=4, R=[8], T=[50], output=MSTEP,
   methods=['fakgd_v1_off', 'fakgd_v1_clamp', 'fakgd_v1_full'])
display(analysis.mstep_report(analysis.load(MSTEP)))
# Inert if mean_rel_change_sigma2 < 0.01 and delta ~ 0  ->  drop the M-step (plan C3).
""")

md(r"""
### 1b — tune the isotropic baselines per (R, T); FA-KGD once at T = 50
""")

code(r"""
TUNE1 = f'{OUT}/g1_tune.jsonl'
TUNED = f'{OUT}/tuned.json'
iso = run.tuning_configs({k: run.DEFAULT_GRIDS[k] for k in ['ikdc_iso', 'pigdm_iso']})
fak = run.tuning_configs({k: run.FAKGD_GRIDS[k] for k in ['ikdc_fakgd', 'pigdm_fakgd']})
go(split='calib', n_slices=4, R=[4, 8], T=[20, 50, 100], methods=iso, output=TUNE1)
go(split='calib', n_slices=4, R=[4, 8], T=[50], methods=fak, output=TUNE1)
tuned = run.select_tuned(TUNE1, TUNED)
print(json.dumps(tuned, indent=1))
""")

md(r"""
### 1c — test (8 volumes × 4 slices) and the gate
""")

code(r"""
G1 = f'{OUT}/g1_test.jsonl'
go(split='test', n_volumes=8, n_slices=4, R=[4, 8], T=[20, 50, 100], tuned=TUNED, output=G1,
   methods=['ikdc_iso', 'ikdc_fakgd', 'pigdm_iso', 'pigdm_fakgd'])
v = vol(G1)
for host in ['ikdc', 'pigdm']:
    g = analysis.gate_g1(v, host=host)
    print(f'\n=== G1 ({host}) ===')
    display(g[['R', 'T', 'mean_b', 'mean_a', 'delta', 'ci', 'n_positive', 'n_volumes', 'p_wilcoxon_one_sided', 'pass']].round(3))
    g.to_csv(f'{OUT}/g1_gate_{host}.csv', index=False)
""")

md(r"""
**If G1 fails:** do not run Stage 2. See the revision plan, "Risks and fallbacks".
Before deciding, check the one cheap rescue: `pigdm_fakgd` vs `pigdm_iso` (the white-spectrum
ΠGDM is the closest baseline in spirit), and the per-volume deltas (`analysis.paired`).
""")

md(r"""
## Stage 2 — Gate G2 (by **1 Nov**): plug-in table, step-robustness, mechanism
### 2a — tune the remaining hosts
""")

code(r"""
TUNE2 = f'{OUT}/g2_tune.jsonl'
iso = run.tuning_configs({k: run.DEFAULT_GRIDS[k] for k in ['dds_iso', 'diffpir_iso', 'dps']})
fak = run.tuning_configs({k: run.FAKGD_GRIDS[k] for k in ['dds_fakgd', 'diffpir_fakgd']})
go(split='calib', n_slices=4, R=[4, 8], T=[10, 20, 50, 100, 200], methods=iso, output=TUNE2)
go(split='calib', n_slices=4, R=[4, 8], T=[50], methods=fak, output=TUNE2)
# merge with stage-1 tuning (also tunes ikdc/pigdm at T=10 and 200 for the step sweep)
iso12 = run.tuning_configs({k: run.DEFAULT_GRIDS[k] for k in ['ikdc_iso', 'pigdm_iso']})
go(split='calib', n_slices=4, R=[4, 8], T=[10, 200], methods=iso12, output=TUNE1)
t1 = run.select_tuned(TUNE1, f'{OUT}/_t1.json'); t2 = run.select_tuned(TUNE2, f'{OUT}/_t2.json')
json.dump({**t1, **t2}, open(TUNED, 'w'), indent=1)
""")

md(r"""
### 2b — main plug-in table (Tab. 3), T = 50, all 24 test volumes
""")

code(r"""
MAIN = f'{OUT}/g2_main_brain.jsonl'
hosts = ['ikdc', 'pigdm', 'dds', 'diffpir']
go(split='test', n_slices=4, R=[4, 8], T=[50], tuned=TUNED, output=MAIN,
   methods=[f'{h}_{s}' for h in hosts for s in ('iso', 'fakgd')] + ['dps'])
v = vol(MAIN)
rows = [analysis.paired(v, f'{h}_fakgd', f'{h}_iso', R=R, T=50) for h in hosts for R in (4, 8)]
display(pd.DataFrame(rows)[['a', 'R', 'mean_b', 'mean_a', 'delta', 'ci', 'n_positive', 'p_wilcoxon_one_sided']].round(3))
""")

md(r"""
### 2c — step sweep (Tab. 4, Fig. 1) and mechanism diagnostics (Fig. 2)
""")

code(r"""
STEPS = f'{OUT}/g2_steps.jsonl'
go(split='test', n_slices=4, R=[8, 4], T=[10, 20, 50, 100, 200], tuned=TUNED, output=STEPS,
   methods=['ikdc_iso', 'ikdc_fakgd', 'pigdm_iso', 'pigdm_fakgd'])
# per-step diagnostics on a 4-volume subset (stored separately: records are larger)
DIAG = f'{OUT}/g2_diag.jsonl'
go(split='test', n_volumes=4, n_slices=4, R=[8], T=[100], tuned=TUNED, output=DIAG, diagnostics=True,
   methods=['pigdm_iso', 'pigdm_fakgd', 'ikdc_iso', 'ikdc_fakgd'])
v = vol(STEPS)
print(analysis.steps_table_rows(v, R=8))       # paste into results.tex, Tab. 4
""")

code(r"""
import matplotlib.pyplot as plt, numpy as np
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
analysis.plot_steps(vol(STEPS), R=8, hosts=['ikdc', 'pigdm'], ax=ax[0]); ax[0].set_title('R = 8')
analysis.plot_steps(vol(STEPS), R=4, hosts=['ikdc', 'pigdm'], ax=ax[1]); ax[1].set_title('R = 4')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_teaser_steps.pdf'); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for m, ls in (('pigdm_iso', '-'), ('pigdm_fakgd', '--')):
    d = analysis.load_diagnostics(DIAG, m, R=8, T=100)
    med_b = np.nanmedian(d['band_nmse'], 0); med_c = np.nanmedian(d['calib_ratio'], 0)
    for b in range(med_b.shape[1]):
        ax[0].semilogy(med_b[:, b], ls, color=f'C{b}', label=f'band {b}' if ls == '-' else None)
        ax[1].plot(med_c[:, b], ls, color=f'C{b}')
ax[0].set_xlabel('reverse step'); ax[0].set_ylabel('k-space NMSE of x0hat'); ax[0].legend(fontsize=7)
ax[1].axhline(1, color='k', lw=0.5); ax[1].set_xlabel('reverse step'); ax[1].set_ylabel('calibration ratio')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_mechanism.pdf'); plt.show()
""")

md(r"""
## Stage 3 — Gate G3 (by **8 Nov**): low SNR (Tab. 5)
White noise is added **after prewhitening** (so it is Ψ-correlated in raw coil space). Noise
variance ×4, ×16, ×64 = SNR ×½, ×¼, ×⅛. M4Raw: see the README (needs its own cohort tarball).
""")

code(r"""
SNR = f'{OUT}/g3_snr.jsonl'
go(split='test', n_volumes=12, n_slices=4, R=[4], T=[50], tuned=TUNED, output=SNR,
   noise_scales=[0.0, 3 ** 0.5, 15 ** 0.5, 63 ** 0.5],
   methods=['ikdc_iso', 'ikdc_fakgd', 'pigdm_iso', 'pigdm_fakgd'])
v = vol(SNR)
rows = [analysis.paired(v, f'{h}_fakgd', f'{h}_iso', R=4, T=50, snr_tag=t)
        for h in ('ikdc', 'pigdm') for t in sorted(v.snr_tag.unique())]
display(pd.DataFrame(rows)[['a', 'snr_tag', 'mean_b', 'mean_a', 'delta', 'ci', 'n_positive']].round(3))
# NOTE: tuned params come from native-SNR calibration; for a fully fair low-SNR row,
# re-tune on calibration with the same noise_scales (costs ~1 h).
""")

md(r"""
## Stage 4 — ablations (Tab. 7) and compute (Tab. 8)
Host defaults to IK-DC (switch `ABL_HOST` to `'pigdm'` if ΠGDM proved stable in Stage 1).
Row (f), the schedule of Tian et al. (2026), is **not implemented yet**.
""")

code(r"""
ABL_HOST = 'ikdc'
b = f'{ABL_HOST}_fakgd'
abl = [f'{ABL_HOST}_iso', b,
       {'name': 'abl_c_gate',     'base': b, 'gate_beta': 1.0},
       {'name': 'abl_d_gateonly', 'base': f'{ABL_HOST}_iso', 'gate_beta': 1.0},
       {'name': 'abl_e_refine',   'base': b, 'refine': True},
       {'name': 'abl_g_oracle',   'base': b, 'spec_mode': 'oracle'},
       {'name': 'abl_h_s2x0.25',  'base': b, 's2_scale': 0.25},
       {'name': 'abl_i_s2x4',     'base': b, 's2_scale': 4.0},
       {'name': 'abl_j_bins',     'base': b, 'spec_mode': 'bins'}]
ABL = f'{OUT}/g4_ablation.jsonl'
go(split='test', n_slices=4, R=[8], T=[50, 100], tuned=TUNED, output=ABL, methods=abl)
v = vol(ABL)
names = [m if isinstance(m, str) else m['name'] for m in abl]
rows = [analysis.paired(v, n, f'{ABL_HOST}_iso', R=8, T=T) for n in names[1:] for T in (50, 100)]
display(pd.DataFrame(rows)[['a', 'T', 'mean_a', 'delta', 'ci', 'n_positive']].round(3))
""")

code(r"""
# compute table (Tab. 8) from the main run
d = analysis.load(MAIN)
c = d[(d.R == 8)].groupby('method').agg(nfe=('nfe', 'mean'), time_s=('runtime_s', 'mean'),
                                        mem_gb=('peak_mem_gb', 'max')).round(2)
display(c)
""")

md(r"""
## Export
Everything is in Drive (`OUT`, synced in API mode). This cell writes a summary JSON that the paper tables
are filled from; commit it to `MIDL2027/paper/generated/` locally.
""")

code(r"""
summary = {}
for name, path in [('g1', G1), ('main', MAIN), ('steps', STEPS), ('snr', SNR), ('ablation', ABL)]:
    if os.path.exists(path):
        summary[name] = vol(path).to_dict(orient='records')
json.dump(summary, open(f'{OUT}/summary_volume_level.json', 'w'))
print('wrote', f'{OUT}/summary_volume_level.json')
sync(verbose=True)
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("midl27_colab.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
