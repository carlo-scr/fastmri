"""Generate uq_pilot_colab.ipynb (run: python build_notebook.py).

The notebook embeds the current uq.py (written to the Colab checkout by a %%writefile cell),
so it runs against the already-pushed branch without pushing this folder first.
Run a *copy* (uq_pilot_colab_run.ipynb, gitignored): rebuilding overwrites outputs.
Cells tagged 'setup' are Colab-only; test_local.py runs every other cell on the CPU.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s, tags=(), hidden=False):
    meta = {"tags": list(tags)} if tags else {}
    if hidden:
        meta["jupyter"] = {"source_hidden": True}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


# =============================================================================
md(r"""
# Exploration pilot: when to measure again, when to measure longer, when not to trust

**Purpose.** About two hours on one A100, to find out which directions show a real signal on *our*
data before we commit to one. This is not a paper experiment. Every threshold that decides
"promising" is fixed in the config cell, before any result is seen.

**Setup.**
- **Prior:** the 3T fastMRI-brain EDM (Ambient-DPS, supervised R = 1).
- **Posterior sampler:** IK-DC (`fdc.py`). Each slice gets K posterior samples plus 2 *hold-out*
  members, each of which misses 10% of the measured lines.
- **Data:** fastMRI AXFLAIR at 3T (`midl27_brain.tar`) and M4Raw at 0.3T (range-fetched).
- **Nested masks:** every acceleration is a prefix of one line order, so "measure again" means
  *adding* lines to the ones already acquired.
- **Errors on M4Raw:** these need no ground truth. The reconstruction uses repetition 0 only, and
  the held-out repetitions are an independent noisy reference whose noise variance is estimated
  and subtracted. Per pixel, that estimate is unbiased for T1/T2, which have 3 repetitions.

| Probe | Question | Decision it informs | Literature gap it tests |
|---|---|---|---|
| **A** uncertainty signal | Does posterior spread predict error, per slice and *inside tissue*, beyond edges? Is it calibrated, and does one scale factor transfer across R, field and contrast? | Everything below rests on this | Diffusion posterior variance is reported as under-covering (Qiu et al. 2026). Variance and error correlate over the whole image but not inside the object (Zhao et al. 2026). Hard-DC samplers have not been benchmarked |
| **B** measure again | Nested re-acquisition R = 8 → 4 → 2 → full, stopping when predicted error is small. What fraction of the oracle's scan-time savings does it capture? With a threshold calibrated by conformal risk control (Learn-then-Test as the stricter variant), does the violation rate hold *at the stopping time*? | Re-scan or stop | Wen, Ahmad & Schniter (ECCV 2024, TMLR 2025) do this with a CNF and no calibrated stopping rule. CUTE-MRI loses coverage at stopping |
| **C** measure longer | At 0.3T, at equal scan time, is it better to acquire more lines or more averages (also variable-density averaging)? Can the uncertainty after one average predict who gains from a second? | NEX per subject | Online, uncertainty-driven NEX selection appears not to exist. NexOP and CS-VDA fix the allocation offline |
| **D** don't trust (OOD) | Under added noise, mid-scan motion, and 3T → 0.3T, which label-free score flags *failures*: ensemble spread, k-space hold-out residual, prior typicality, or SNR? Does a threshold calibrated in-distribution stay valid under shift? | Runtime monitor | Held-out-line residuals (Ward 2009; SSDU) have not been tested as monitors for diffusion samplers. OOD is not the same as failure |
| **E** small findings | Are small synthetic lesions kept at R = 8? When one is lost, does uncertainty rise there, or is the loss silent? | Lesion-level re-scan trigger | Hallucination and loss of findings (fastMRI 2020 challenge; Bhadra et al. 2021) |
| **F** frequency-aware (FA-KGD) | Does spectral DC give *better-calibrated* ensembles, and does the analytic per-frequency (Wiener) variance give useful uncertainty for free? | FA-KGD as a UQ method rather than a PSNR method | Free Tweedie or Kalman covariance has not been compared against ensembles on raw MRI |
| **G** where to measure | Do max-variance lines beat the next nested lines at the same budget? | Active sampling (links to PVAS) | Bakker et al. 2022: adaptive policies collapse to non-adaptive |
| **H** (optional) adapted prior | If the repetition pilot's post-trained prior exists: does adaptation fix calibration on 0.3T? | Monitor → adapt loop | — |

**How to read the outcome.** The last cell prints a scoreboard. Each row is PROMISING / WEAK / NO SIGNAL
against the pre-set criteria, next to the paper it would support. A "no signal" is useful too: it rules
a direction out cheaply. SISL framing: each decision is a stopping or allocation problem under a
chance constraint, with the monitor threshold set by Learn-then-Test / conformal risk control
(cf. ConstrainedZero, Moss et al. IJCAI 2024).

**Before running.** You need an A100 runtime. `network-snapshot.pkl` and `midl27_brain.tar` must be
in `MyDrive/fastmri_artifacts/` (both were used before). Branch `research/sketchsense-wacv27` must be
on GitHub (it is). `uq.py` is embedded below, so this folder does not need to be pushed. Results go
to `fastmri_artifacts/uq_pilot_results/`. Every stage resumes after a disconnect.
""")

code(r"""
# 0.1  config. Every threshold that decides 'promising' is fixed here, before any result.
BRANCH = 'research/sketchsense-wacv27'
ADPS_COMMIT = '75feb2fad4b53e48791fcb0257f2e53a25340774'
DEVICE, SEED = 'cuda', 2027
BUDGET_MIN = 120                      # whole notebook on an A100; cell 0.5 shrinks the cohort to fit
T, K, N_FOLDS = 30, 8, 2              # sampler steps, posterior samples, k-space hold-out members
FASTMRI_TAR = 'midl27_brain.tar'      # 28 AXFLAIR volumes x 8 central slices
N_FASTMRI_VOLS, FASTMRI_SLICES, FASTMRI_RS = 28, (2, 5), (8, 4, 2)
N_M4RAW_SUBJ, M4RAW_CONTRASTS, M4RAW_PICK, M4RAW_RS = 10, ('T1', 'T2', 'FLAIR'), (1, 4), (4, 2)
KAPPA = {('fastmri', 8): 3.0, ('fastmri', 4): 1.0, ('fastmri', 2): 1.0,   # tuned at T=50 (G1)
         ('m4raw', 4): 1.0, ('m4raw', 2): 1.0, ('m4raw', 1): 1.0}
KAPPA_SPEC = 3.0                      # FA-KGD (spectral) DC, tuned at T=50 (G1)
NOISE_VAR = 15.0                      # D: added noise -> total variance x16 (SNR / 4)
N_G = 16                              # G: slices
ALPHA, DELTA = 0.10, 0.10             # B/D: target violation rate, Learn-then-Test confidence
RUN = dict(A=True, C=True, D=True, E=True, F=True, G=True, H=True)

# (name, promising, weak, higher_is_better). Literature anchors in the intro table.
CRITERIA = {
    'A1': ('slice ranking: Spearman(pred, NMSE) within R, median over groups', 0.6, 0.3, True),
    'A2': ('localisation beyond edges: patch AUROC(pred) - AUROC(edge), median', 0.05, 0.0, True),
    'A3': ('one scale for all R/field: max |90% coverage - 0.90| across groups', 0.05, 0.15, False),
    'B':  ('measure again: fraction of oracle scan-time savings at 10% violation (fastMRI R8->4->2->full)', 0.5, 0.2, True),
    'B_m4': ('measure again at 0.3T: same, M4Raw T1/T2 (R4 -> R2 -> R1)', 0.5, 0.2, True),
    'B_crc': ('calibrated stop (CRC): P(test violation > alpha + 0.05) over unit splits', 0.1, 0.25, False),
    'C1': ('lines vs averages: equal-time win rate of the better trade', 0.7, 0.55, True),
    'C2': ('adaptive NEX: best Spearman(predictor, realised gain of 2nd average)', 0.5, 0.3, True),
    'D1': ('failure detection under shift: best label-free pooled AUROC', 0.85, 0.7, True),
    'D2': ('monitor validity: worst accepted-violation / alpha across shifts (best score)', 2.0, 4.0, False),
    'E':  ('lesion trigger: AUROC(uncertainty feature -> lesion lost)', 0.75, 0.6, True),
    'F1': ('FA-KGD calibration: reduction of |log10 calib ratio| (iso - spectral), R8', 0.1, 0.03, True),
    'F2': ('free analytic UQ: Spearman(analytic, NMSE) / Spearman(ensemble, NMSE)', 0.8, 0.5, True),
    'G':  ('where: max-variance vs nested next lines, median dB gain', 0.5, 0.2, True),
    'H':  ('post-trained prior on 0.3T: median dB gain over zero-shot at R4', 1.0, 0.3, True),
}
""")

code(r"""
# 0.2  code and dependencies (Colab)
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
os.makedirs('/content/fastmri/experiments/uq_pilot', exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""", tags=("setup",))

uq_src = (HERE / "uq.py").read_text()
code("%%writefile /content/fastmri/experiments/uq_pilot/uq.py\n" + uq_src, tags=("setup",), hidden=True)

code(r"""
# 0.3  storage, prior, data locations (Colab)
import sys, json, time, shutil, tarfile, copy
for p in ('/content/fastmri', '/content/fastmri/experiments/uq_pilot', '/content/fastmri/experiments/midl27_reps'):
    sys.path.insert(0, p)
import numpy as np, torch, pandas as pd
import importlib, uq, m4raw as M, storage
uq = importlib.reload(uq)                 # picks up a re-written uq.py without a restart
from src.models.edm_loader import load_edm_model
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
T_START = time.time()

D = storage.Drive(results='uq_pilot_results', local_out='/content/uq_pilot_results')
OUT, RAW, MAPS = D.out, '/content/m4raw_raw', '/content/uq_maps'
sync = D.sync
CK = 'checkpoints/edm/supervised_R=1'
D.get_input('network-snapshot.pkl', f'{CK}/network-snapshot.pkl')
base_net = load_edm_model(CK, method='auto', device=DEVICE).eval().requires_grad_(False)
den = uq.make_denoiser(base_net)

def other_result(folder, name, dst):
    # one file from another results folder on Drive (the repetition pilot's), or None
    if D.mode == 'mount':
        src = f'{D.art}/{folder}/{name}'
        return shutil.copy(src, dst) if os.path.exists(src) else None
    f = D._find(folder, D.art_id, folder=True)
    g = D._find(name, f['id']) if f else None
    if not g:
        return None
    D._download(g['id'], dst)
    return dst

FASTMRI_DIR = '/content/data/midl27_cohort'
if not os.path.isdir(FASTMRI_DIR):
    tar = D.get_input(FASTMRI_TAR, f'/content/{FASTMRI_TAR}')
    with tarfile.open(tar) as t:
        t.extractall('/content/data', filter='data')
    os.remove(tar)
lst_val = uq.retry(lambda: M.zip_listing('val', cache=f'{OUT}/listing_val.json'))
cat_val = M.catalog(lst_val)
sp = other_result('midl27_reps_results', 'splits.json', '/content/reps_splits.json')
EXCLUDE = set(json.load(open(sp))['eval']) if sp else set()      # keep the MIDL eval subjects unseen
FT_PATH = other_result('midl27_reps_results', 'p2_ft_ema.pt', '/content/p2_ft_ema.pt') if RUN['H'] else None
print(f'fastMRI volumes: {len([f for f in os.listdir(FASTMRI_DIR) if f.endswith(".h5")])} | M4Raw val subjects: '
      f'{len(cat_val)} ({len(EXCLUDE)} excluded: repetition-pilot eval split) | post-trained prior: {bool(FT_PATH)}')
""", tags=("setup",))

code(r"""
# 0.4  cases (slices): fastMRI 3T AXFLAIR and M4Raw 0.3T T1/T2/FLAIR
t0 = time.time()
vols = sorted(f for f in os.listdir(FASTMRI_DIR) if f.endswith('.h5'))[:N_FASTMRI_VOLS]
CF = [c for v in vols for c in uq.load_fastmri(os.path.join(FASTMRI_DIR, v), FASTMRI_SLICES)]
print(f'fastMRI: {len(CF)} slices from {len(vols)} volumes ({(time.time() - t0) / 60:.1f} min)')
t0 = time.time()
subs = [s for s in sorted(cat_val) if s not in EXCLUDE and all(c in cat_val[s] for c in M4RAW_CONTRASTS)]
subs = sorted(np.random.default_rng(SEED).choice(subs, min(N_M4RAW_SUBJ, len(subs)), replace=False).tolist())
uq.prefetch('val', [n for s in subs for c in M4RAW_CONTRASTS for n in cat_val[s][c]], RAW, lst_val)
CM = [c for s in subs for ct in M4RAW_CONTRASTS
      for c in uq.load_m4raw('val', s, ct, cat_val, lst_val, RAW, pick=M4RAW_PICK)]
print(f'M4Raw: {len(CM)} slices from {len(subs)} subjects ({(time.time() - t0) / 60:.1f} min)')
cases_df = pd.DataFrame([dict(key=c.key, dataset=c.dataset, unit=c.unit, contrast=c.contrast, slice=c.slice,
                              n_reps=c.k.shape[0], coils=c.k.shape[1], acq_cols=int(c.acq.sum()),
                              snr=float(np.sqrt(np.mean(np.abs(c.ref[c.region]) ** 2) / c.v_rep)),
                              physio_frac=(c.v_rep / c.meta['nv1_mean'] - 1) if 'nv1_mean' in c.meta else np.nan)
                         for c in CF + CM])
cases_df.to_csv(f'{OUT}/cases.csv', index=False); sync()
display(cases_df.groupby(['dataset', 'contrast'])[['snr', 'physio_frac', 'coils', 'acq_cols']].median().round(3))
""")

code(r"""
# 0.5  cost of one ensemble on this GPU -> fit the cohort into the budget
KS = min(4, K)                        # small ensembles (C, G) and the matched prefix of A's ensembles
from src.samplers.schedules import edm_sigma_schedule
SIG = edm_sigma_schedule(T, sigma_max=10.0).to(DEVICE)

def bench(case, reps=2, **mkw):
    meas = uq.build_measurement(case, device=DEVICE, **mkw)
    uq.sample_batch(den, meas['op'], meas['y'], meas['s2'], SIG[:3], spectrum=meas['spectrum'])   # warm-up
    if DEVICE == 'cuda': torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        uq.sample_batch(den, meas['op'], meas['y'], meas['s2'], SIG, spectrum=meas['spectrum'])
    if DEVICE == 'cuda': torch.cuda.synchronize()
    return 1.15 * (time.time() - t) / reps            # +15% for summaries and typicality

tf, tf8, tf4 = bench(CF[0], R=4, K=K, n_folds=N_FOLDS), bench(CF[0], R=8, K=K), bench(CF[0], R=4, K=KS)
tm, tm4 = bench(CM[0], R=4, K=K, n_folds=N_FOLDS), bench(CM[0], R=4, K=KS)
print(f'seconds per ensemble: fastMRI {tf:.1f} (B={K + N_FOLDS}), {tf8:.1f} (B={K}), {tf4:.1f} (B={KS}) | '
      f'M4Raw {tm:.1f} (B={K + N_FOLDS}), {tm4:.1f} (B={KS})')

def first_per_unit(cases):
    seen, out = set(), []
    for c in cases:
        if c.unit not in seen:
            seen.add(c.unit); out.append(c)
    return out

def projection(nf, nm):
    cf = nf * len(FASTMRI_SLICES); cm = nm * len(M4RAW_CONTRASTS) * len(M4RAW_PICK)
    cm12 = nm * sum(c in ('T1', 'T2') for c in M4RAW_CONTRASTS) * len(M4RAW_PICK)
    p = {'A': cf * len(FASTMRI_RS) * tf + cm * len(M4RAW_RS) * tm, 'C': cm12 * 7 * tm4, 'D': nf * 2 * tf,
         'E': nf * 2 * tf8, 'F': nf * tf, 'G': min(N_G, nf) * 2 * tf4, 'H': cm * tm if FT_PATH else 0}
    return {k: v / 60 for k, v in p.items() if RUN[k]}

UF, UM = list(dict.fromkeys(c.unit for c in CF)), list(dict.fromkeys(c.unit for c in CM))
nf, nm = len(UF), len(UM)
avail = BUDGET_MIN - (time.time() - T_START) / 60 - 8          # keep ~8 min for the analysis cells
while sum(projection(nf, nm).values()) > avail and (nf > 8 or nm > 4):
    nf, nm = (nf - 2 if nf > 8 else nf), (nm - 1 if nm > 4 else nm)
CF = [c for c in CF if c.unit in set(UF[:nf])]; CM = [c for c in CM if c.unit in set(UM[:nm])]
CF1 = first_per_unit(CF)                                        # one slice per volume (D, E, F, G)
CM12 = [c for c in CM if c.k.shape[0] >= 3]                     # >= 3 repetitions (C, M4Raw cascade)
proj = projection(nf, nm)
print(f'plan: {nf} fastMRI volumes ({len(CF)} slices), {nm} M4Raw subjects ({len(CM)} slices, {len(CM12)} with 3 reps)')
print('projected minutes:', {k: round(v, 1) for k, v in proj.items()}, '| total', round(sum(proj.values()), 1),
      '| available', round(avail, 1))
RES = f'{OUT}/records.jsonl'
run = uq.Runner(RES, den, T=T, device=DEVICE, maps_dir=MAPS, sync=sync)
""")

# =============================================================================
md(r"""
## Runs (each cell resumes; records go to `records.jsonl`)
Paired design: within a slice, every variant (shift, spectral DC, lesion) reuses the same mask,
hold-out folds and noise seed as its reference run. Differences are therefore due to the variant only.
""")

code(r"""
# A  posterior ensembles on nested masks: fastMRI R = 8, 4, 2 and M4Raw R = 4, 2 (typicality at R = 4)
if RUN['A']:
    ex = {}
    for i, c in enumerate(CF):
        for R in FASTMRI_RS:
            ex[R] = ex.get(R, 0) + 1
            run.run('A', c, f'R{R}', mkw=dict(R=R, K=K, n_folds=N_FOLDS), kappa=KAPPA[('fastmri', R)],
                    skw=dict(typ=(R == 4)), maps=ex[R] <= 2)
        if (i + 1) % 8 == 0: run.progress(f'A fastMRI {i + 1}/{len(CF)}')
    ex = {}
    for i, c in enumerate(CM):
        for R in M4RAW_RS:
            ex[R] = ex.get(R, 0) + 1
            run.run('A', c, f'R{R}', mkw=dict(R=R, K=K, n_folds=N_FOLDS), kappa=KAPPA[('m4raw', R)],
                    skw=dict(xs_last=c.xs[-1], typ=(R == 4)), maps=ex[R] <= 2)
        if (i + 1) % 8 == 0: run.progress(f'A M4Raw {i + 1}/{len(CM)}')
    run.progress('A done')
""")

code(r"""
# C  measure longer: lines vs averages at 0.3T (T1/T2, 3 repetitions; reference = repetition 2 only)
#    alloc = number of averages per phase-encode column; 'vda' averages the central half of the sampled lines
def alloc_for(c, R, scheme):
    W = c.k.shape[-1]; cols = c.columns(R); a = np.zeros(W, int); a[cols] = 1
    if scheme == 'nex2':
        a[cols] = 2
    elif scheme == 'vda':
        d = np.abs(cols - W // 2); a[cols[d <= np.quantile(d, 0.5)]] = 2
    return a

C_CONFIGS = [(1, 'nex1'), (1, 'nex2'), (1, 'vda'), (2, 'nex2'), (2, 'vda'), (4, 'nex2'), (4, 'vda')]
if RUN['C']:
    for i, c in enumerate(CM12):
        for R, scheme in C_CONFIGS:
            run.run('C', c, f'R{R}_{scheme}', mkw=dict(alloc=alloc_for(c, R, scheme), K=KS), kappa=KAPPA[('m4raw', R)],
                    skw=dict(ref=c.xs[-1], ref_nv=np.full(c.ref.shape, c.v_rep, np.float32), xs_last=c.xs[-1],
                             detail=False), extra=dict(scheme=scheme, R_nom=R))
        if (i + 1) % 8 == 0: run.progress(f'C {i + 1}/{len(CM12)}')
    run.progress('C done')
""")

code(r"""
# D  shifts at R = 4 (paired with A): added noise (SNR / 4) and rigid motion mid-scan
if RUN['D']:
    for i, c in enumerate(CF1):
        mk = dict(R=4, K=K, n_folds=N_FOLDS)
        run.run('D', c, 'noise', mkw=dict(mk, noise_var=NOISE_VAR), kappa=KAPPA[('fastmri', 4)], skw=dict(typ=True),
                extra=dict(shift='noise'))
        mo = uq.random_motion(uq.stable_seed(c.key, 'motion'))
        run.run('D', c, 'motion', mkw=dict(mk, motion=mo), kappa=KAPPA[('fastmri', 4)], skw=dict(typ=True),
                extra=dict(shift='motion', motion_amp=mo['amp'], motion_t=mo['t']))
    run.progress('D done')
""")

code(r"""
# E  two small synthetic lesions per slice at R = 8: clean vs lesion run, same seed (paired)
if RUN['E']:
    for i, c in enumerate(CF1):
        if run.has('E', c, 'metrics'):
            continue
        blob, les = uq.make_lesions(c, seed=uq.stable_seed(c.key, 'lesion'))
        if not les:
            continue
        mk = dict(R=8, K=K)
        _, (Xc, _) = run.run('E', c, 'clean', mkw=mk, kappa=KAPPA[('fastmri', 8)], skw=dict(detail=False), keep=True)
        _, (Xl, _) = run.run('E', c, 'lesion', mkw=dict(mk, lesion=blob), kappa=KAPPA[('fastmri', 8)],
                             skw=dict(detail=False), keep=True)
        lm = uq.lesion_metrics(c, Xl, Xc, K, blob, les)
        run.write(dict(rid=run.rid('E', c, 'metrics'), probe='E', key=c.key, unit=c.unit, dataset=c.dataset,
                       tag='metrics', lesions=lm))
        if i < 2:
            os.makedirs(MAPS, exist_ok=True)
            np.savez_compressed(f'{MAPS}/E_example_{i}.npz', ref=c.ref, blob=blob, region=c.region,
                                cln=Xc[:K].mean(0).cpu().numpy(), les=Xl[:K].mean(0).cpu().numpy(),
                                centers=np.array([[l['cy'], l['cx']] for l in les]))
    run.progress('E done')
""")

code(r"""
# F  FA-KGD (spectral DC) ensembles at R = 8, paired with A's isotropic run
if RUN['F']:
    for c in CF1:
        run.run('F', c, 'R8_spectral', mkw=dict(R=8, K=K, n_folds=N_FOLDS), cov='spectral', kappa=KAPPA_SPEC)
    run.progress('F done')
""")

code(r"""
# G  where to measure: from R = 8 add as many lines as R = 4 would, chosen by ensemble k-space variance
#    or at random; compare with A's nested R = 4 (same budget). K = KS everywhere (A's matching prefix).
if RUN['G']:
    a = uq.load_jsonl(RES)
    r8 = a[(a.probe == 'A') & (a.tag == 'R8') & (a.dataset == 'fastmri')].set_index('key')
    for c in CF1[:N_G]:
        if c.key not in r8.index:
            continue
        cv = np.asarray(r8.loc[c.key, 'col_coilvar'], float)
        c8 = c.columns(8)
        n_add = uq.n_lines(c.acq, 4) - len(c8)
        cand = np.setdiff1d(np.flatnonzero(c.acq), c8)
        top = cand[np.argsort(-cv[cand])[:n_add]]
        rnd = np.random.default_rng(uq.stable_seed(c.key, 'G')).choice(cand, n_add, replace=False)
        for tag, cols in (('var', np.r_[c8, top]), ('rand', np.r_[c8, rnd])):
            run.run('G', c, tag, mkw=dict(cols=cols, K=KS), kappa=KAPPA[('fastmri', 4)], skw=dict(detail=False))
    run.progress('G done')
""")

code(r"""
# H  (optional) the repetition pilot's post-trained 0.3T prior (two-regime: post-trained above the noise floor)
if RUN['H'] and FT_PATH:
    ft_net = copy.deepcopy(base_net)
    ft_net.load_state_dict(torch.load(FT_PATH, map_location=DEVICE))
    ft_net.eval().requires_grad_(False)
    for c in CM:
        den_ft = uq.make_denoiser(ft_net, below=base_net, floor=c.meta['floor'])
        run.run('H', c, 'R4_ft', mkw=dict(R=4, K=K, n_folds=N_FOLDS), kappa=KAPPA[('m4raw', 4)],
                skw=dict(xs_last=c.xs[-1]), den_fn=den_ft)
    run.progress('H done')
else:
    print('H skipped (no p2_ft_ema.pt from the repetition pilot)')
""")

# =============================================================================
md(r"""
## Analysis
`pred` is the ensemble-predicted MSE of the posterior mean, (1 + 1/K) × the sample variance. It is
normalised by the slice's signal power (`pred_nmse`) to compare with the actual NMSE.
Calibration ratio = actual / predicted (1 = calibrated; > 1 = overconfident).
""")

code(r"""
# A.1  does the spread predict the error?  (unit for CIs: volume / subject)
import matplotlib.pyplot as plt
res = uq.load_jsonl(RES)
A = res[res.probe == 'A'].copy()
A['group'] = A.dataset + ' R' + A.R.astype(int).astype(str)
A['pred_k2n'], A['nmse_k2'] = A.pred_k2 / A.sig_pow, A.mse_k2 / A.sig_pow
rho = lambda d, a, b: uq._spearman(d[a], d[b])
rows = []
for g, d in A.groupby('group'):
    rows.append(dict(group=g, n=len(d), units=d.unit.nunique(), nmse_med=d.nmse.median(), psnr_med=d.psnr.median(),
                     rho_ens=rho(d, 'pred_nmse', 'nmse'),
                     rho_ens_ci=uq.boot_ci(d, 'unit', lambda x: rho(x, 'pred_nmse', 'nmse'), n=300),
                     rho_ens_k2=rho(d, 'pred_k2n', 'nmse_k2'), rho_holdout=rho(d, 'cv_nmse', 'nmse'),
                     rho_analytic=rho(d, 'pred_ana_nmse', 'nmse'), rho_inv_snr=uq._spearman(-d.snr_data, d.nmse),
                     calib_ratio=(d.mse / d.pred_mse).median(),
                     pix_rho_pred=d.pix_rho_pred.median(), pix_rho_edge=d.pix_rho_edge.median(),
                     patch_auc_pred=d.patch_auc_pred.median(), patch_auc_edge=d.patch_auc_edge.median(),
                     patch_auc_int=d.patch_auc_int.median(),
                     holdout_cost_db=float(np.median(10 * np.log10(np.maximum(d.mse_holdout, 1e-12) / np.maximum(d.mse_single, 1e-12)))),
                     disp_meas=d.disp_meas.median()))
tabA = pd.DataFrame(rows).set_index('group')
display(tabA.round(3))
m4 = A[A.dataset == 'm4raw']
if len(m4):
    print('M4Raw by contrast (Spearman pred vs NMSE, calibration ratio):')
    display(m4.groupby(['R', 'contrast']).apply(lambda d: pd.Series(dict(n=len(d), rho_ens=rho(d, 'pred_nmse', 'nmse'),
                                                                            calib=(d.mse / d.pred_mse).median()))).round(3))
""")

code(r"""
# A.2  figures: predicted vs actual NMSE per slice; patch reliability diagram; example maps
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for g, d in A.groupby('group'):
    ax[0].loglog(d.pred_nmse, d.nmse, 'o', ms=3, alpha=.7, label=g)
lim = [A[['pred_nmse', 'nmse']].clip(lower=1e-6).min().min(), A[['pred_nmse', 'nmse']].max().max()]
ax[0].plot(lim, lim, 'k--', lw=.8); ax[0].set_xlabel('predicted NMSE (ensemble)'); ax[0].set_ylabel('actual NMSE')
ax[0].legend(fontsize=7); ax[0].set_title('slice level')
for g, d in A.groupby('group'):
    n, p, e = (np.sum(np.stack(d[c].to_numpy()), 0) for c in ('rel_n', 'rel_pred', 'rel_err'))
    ok = n >= 30
    ax[1].loglog(p[ok] / n[ok], np.maximum(e[ok] / n[ok], 1e-8), '.-', label=g)
ax[1].plot([1e-6, 1], [1e-6, 1], 'k--', lw=.8); ax[1].set_xlabel('predicted patch MSE / signal')
ax[1].set_ylabel('actual patch MSE / signal'); ax[1].set_title('8x8 patches (reliability)'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_A_calibration.png', dpi=130); plt.show()

ex_files = sorted(f for f in os.listdir(MAPS) if f.startswith('A_')) if os.path.isdir(MAPS) else []
pick = [f for f in ex_files if 'fastmri' in f and '_R8_' in f][:1] + [f for f in ex_files if 'm4raw' in f and '_R4_' in f][:1]
if pick:
    fig, ax = plt.subplots(len(pick), 4, figsize=(13, 3.4 * len(pick)), squeeze=False)
    for r, f in enumerate(pick):
        z = np.load(f'{MAPS}/{f}')
        reg = z['region']                                 # outside the coil-map support the data say nothing
        ims = [np.abs(z['ref']), np.abs(z['xbar']), np.sqrt(np.maximum(z['err'], 0)) * reg, np.sqrt(z['pred']) * reg]
        vmax = np.quantile(ims[0][reg], 0.99)
        for j, (im, t) in enumerate(zip(ims, ['reference', 'posterior mean', '|error| in region (noise-corrected)',
                                              'posterior std in region (same scale)'])):
            ax[r, j].imshow(im, cmap='gray', vmax=vmax if j < 2 else np.quantile(ims[2][z['region']], 0.99))
            ax[r, j].set_title(f'{f.split("_")[1]} {t}' if j == 0 else t, fontsize=8); ax[r, j].axis('off')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig_A_examples.png', dpi=110); plt.show()
""")

code(r"""
# A.3  calibration transfer: one 90% threshold (from fastMRI R8, half of the volumes) applied everywhere
#      score per pixel = |xbar - ref|^2 / (pred + ref_var); calibrated complex Gaussian -> q90 = log10(ln 10) = 0.362
units8 = sorted(A[A.group == 'fastmri R8'].unit.unique()); cal_u = set(units8[::2])
cal = A[(A.group == 'fastmri R8') & A.unit.isin(cal_u)]
thr = uq.quantile_from_hist(np.sum(np.stack(cal.cov_hist.to_numpy()), 0), uq.COV_EDGES, 0.9)
rows = []
for g, d in A.groupby('group'):
    d = d[~((d.group == 'fastmri R8') & d.unit.isin(cal_u))]
    h = np.sum(np.stack(d.cov_hist.to_numpy()), 0)
    rows.append(dict(group=g, q90_log10=uq.quantile_from_hist(h, uq.COV_EDGES, 0.9),
                     coverage_at_transferred_thr=uq.coverage_from_hist(h, uq.COV_EDGES, thr)))
tabA3 = pd.DataFrame(rows).set_index('group')
tabA3['required_scale'] = 10 ** (tabA3.q90_log10 - np.log10(np.log(10)))
print(f'threshold from fastMRI R8 calibration half: log10 score <= {thr:.2f} (calibrated would be 0.36)')
display(tabA3.round(3))
""")

code(r"""
# A.4  frequency view (per phase-encode column): is the error where the ensemble / analytic variance says?
cs = pd.DataFrame([dict(group=r.group, **uq.column_stats(r)) for _, r in A.iterrows()])
display(cs.groupby('group').median().round(3))
print('ratio_* = actual / predicted on unmeasured or measured columns (1 = calibrated); '
      'rho_ens vs rho_dist: does the ensemble rank unmeasured columns better than distance from the centre?')
""")

code(r"""
# B  measure again: nested cascade R8 -> R4 -> R2 -> full (fastMRI) and R4 -> R2 -> R1 (M4Raw T1/T2)
def wide(df, level_col, levels, cols):
    out = None
    for L in levels:
        d = df[df[level_col] == L].drop_duplicates('key').set_index('key')
        d = d[list(cols.values())].rename(columns={v: f'{k}_{L}' for k, v in cols.items()})
        out = d if out is None else out.join(d, how='inner')
    return out

SCORES = {'pred': 'ensemble (K=8)', 'k2': 'ensemble (K=2)', 'cv': 'k-space hold-out', 'ana': 'analytic (FA-KGD)'}
f = A[A.dataset == 'fastmri'].copy()
Wf = wide(f, 'R', list(FASTMRI_RS), dict(err='nmse', pred='pred_nmse', k2='pred_k2n', cv='cv_nmse', ana='pred_ana_nmse',
                                         cost='cost'))
first = f.drop_duplicates('key').set_index('key')
Wf['err_full'] = (first.v_rep / first.sig_pow).reindex(Wf.index)     # a fully sampled scan: thermal noise only
Wf['cost_full'] = 1.0
for s in SCORES: Wf[f'{s}_full'] = -np.inf
Wf['unit'] = first.unit.reindex(Wf.index)
LV_F = list(FASTMRI_RS) + ['full']

Wm = None
c_ = res[res.probe == 'C']
m = A[(A.dataset == 'm4raw') & A.key.isin(c_.key)].copy()
if len(m) and len(c_):
    P_case = m[m.R == 4].drop_duplicates('key').set_index('key').sig_pow
    r1 = c_[c_.tag == 'R1_nex1'].assign(R=1)
    mm = pd.concat([m, r1])
    mm['P'] = mm.key.map(P_case)
    mm['err'] = (mm[f'cr_k{KS}'] - mm.v_rep) / mm.P
    mm['pred4'] = mm[f'pred_k{KS}'] / mm.P
    Wm = wide(mm, 'R', [4, 2, 1], dict(err='err', pred='pred4', cost='cost'))
    Wm['unit'] = mm.drop_duplicates('key').set_index('key').unit.reindex(Wm.index)

def b_report(Wd, levels, scores, name):
    # primary tau: median NMSE at the second level (cheapest level mostly fails, the third mostly passes);
    # secondary: median at the first level (a pure accept / re-scan decision on the cheapest level)
    taus = dict(primary=float(Wd[f'err_{levels[1]}'].median()), secondary=float(Wd[f'err_{levels[0]}'].median()))
    rows = []
    for tn, tau in taus.items():
        fixed = uq.fixed_points(Wd, levels, tau)
        print(f'{name}, {tn} tau = {tau:.4g}: fixed protocols', fixed.round(3).to_dict(orient='records'))
        rows += [dict(tau=tn, score=SCORES.get(s, s), **uq.savings_captured(Wd, levels, tau, v, s))
                 for s in scores for v in (0.05, 0.10)]
    tb = pd.DataFrame(rows); display(tb.round(3))
    tau = taus['primary']
    fixed = uq.fixed_points(Wd, levels, tau)
    lt = {}
    for meth in ('crc', 'ltt'):
        lt[meth] = uq.ltt_cascade(Wd, Wd.unit.to_numpy(), levels, tau, ALPHA, DELTA, score='pred', n_splits=200,
                                  method=meth)
        print(f'{meth.upper()} stop (alpha={ALPHA}, 50/50 unit splits, primary tau): test violation mean '
              f'{lt[meth].test_viol.mean():.3f}, P(viol > alpha + 0.05) = {(lt[meth].test_viol > ALPHA + 0.05).mean():.2f}, '
              f'test cost {lt[meth].test_cost.mean():.3f} vs fixed protocol at alpha {uq.timeshare_cost(fixed, ALPHA):.3f}')
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in scores:
        cc = uq.cascade_curve(Wd, levels, tau, s); ax.plot(cc.cost, cc.viol, '.-', ms=3, label=SCORES.get(s, s))
    oc = uq.oracle_frontier(Wd, levels, tau); ax.plot(oc.cost, oc.viol, 'k:', label='oracle')
    ax.plot(fixed.cost, fixed.viol, 'ks-', label='fixed R (time-share between)')
    ax.set_xlabel('mean scan time (fraction of full)'); ax.set_ylabel(f'violation rate (NMSE > {tau:.3g})')
    ax.set_ylim(-0.02, 0.8); ax.legend(fontsize=7); ax.set_title(name)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig_B_{name.split()[0]}.png', dpi=130); plt.show()
    return tb, lt

tbB_f, ltB_f = b_report(Wf, LV_F, list(SCORES), 'fastMRI 3T')
tbB_m, ltB_m = b_report(Wm, [4, 2, 1], ['pred'], 'M4Raw 0.3T') if Wm is not None and len(Wm) >= 6 else (None, None)
""")

code(r"""
# C  measure longer: lines vs averages at equal scan time (0.3T), and who gains from a 2nd average
CC = None
if len(c_):
    a12 = A[(A.dataset == 'm4raw') & A.key.isin(c_.key) & A.R.isin([2, 4])].assign(scheme='nex1', R_nom=lambda d: d.R)
    allc = pd.concat([a12, c_])
    allc['P'] = allc.key.map(P_case)
    allc['err'] = (allc[f'cr_k{KS}'] - allc.v_rep) / allc.P
    allc['cfg'] = 'R' + allc.R_nom.astype(int).astype(str) + '_' + allc.scheme
    CC = allc.pivot_table(index='key', columns='cfg', values='err')
    cost = allc.groupby('cfg').cost.mean()
    byk = {c.key: c for c in CM12}
    mv = pd.DataFrame([dict(key=k, mvue_nex1=(np.mean(np.abs(c.xs[0] - c.xs[-1])[c.region] ** 2) - c.v_rep) / P_case[k],
                            mvue_nex2=(np.mean(np.abs(0.5 * (c.xs[0] + c.xs[1]) - c.xs[-1])[c.region] ** 2) - c.v_rep) / P_case[k])
                       for k, c in byk.items() if k in P_case.index]).set_index('key')
    CC = CC.join(mv)
    cost['mvue_nex1'], cost['mvue_nex2'] = 1.0, 2.0
    vr = allc.drop_duplicates('key').set_index('key')
    ena = (vr.v_rep / vr.P).reindex(CC.index).to_numpy()[:, None] / CC.clip(lower=1e-9)   # equivalent number of averages
    summ = pd.DataFrame(dict(cost=cost, nmse_median=CC.median(), ena_median=ena.median())).sort_values('cost')
    display(summ.round(3))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for fam, mk in (('nex1', 'o-'), ('nex2', 's-'), ('vda', '^-')):
        cs_ = [c for c in summ.index if c.endswith(fam) and c.startswith('R')]
        cs_ = sorted(cs_, key=lambda c: cost[c]); ax.plot(cost[cs_], summ.nmse_median[cs_], mk, label=f'prior, {fam}')
    ax.plot(cost[['mvue_nex1', 'mvue_nex2']], summ.nmse_median[['mvue_nex1', 'mvue_nex2']], 'kx--', label='no prior (full, MVUE)')
    ax.set_xlabel('scan time (fraction of one full average)'); ax.set_ylabel('NMSE (noise-corrected, median)')
    ax.set_yscale('log'); ax.legend(fontsize=7); ax.set_title('0.3T: lines vs averages')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig_C_frontier.png', dpi=130); plt.show()
    pairs = [('R2_nex2', 'R1_nex1'), ('R2_vda', 'R1_nex1'), ('R4_nex2', 'R2_nex1'), ('R4_vda', 'R2_nex1'),
             ('R1_nex1', 'mvue_nex1'), ('R1_nex2', 'mvue_nex2')]
    rows = []
    for a_, b_ in pairs:
        if a_ in CC and b_ in CC:
            d = CC[[a_, b_]].dropna()
            rows.append(dict(alternative=a_, baseline=b_, cost_alt=cost[a_], cost_base=cost[b_],
                             win_rate=float((d[a_] < d[b_]).mean()),
                             db_gain=float(10 * np.log10(d[b_].clip(lower=1e-9).mean() / d[a_].clip(lower=1e-9).mean()))))
    tabC1 = pd.DataFrame(rows); display(tabC1.round(3))
    # who gains from a second average?  predictors come from the 1-average run only
    one = allc[allc.scheme == 'nex1'].set_index(['key', 'R_nom'])
    rows = []
    for R in (1, 2, 4):
        if f'R{R}_nex2' not in CC or f'R{R}_nex1' not in CC:
            continue
        g = (CC[f'R{R}_nex1'] - CC[f'R{R}_nex2']).dropna()
        o = one.xs(R, level='R_nom').reindex(g.index)
        pk = o[f'pred_k{KS}']
        preds = dict(ensemble=pk / o.P, analytic_voi=o.voi_nex, inv_snr=-o.snr_data, ensemble_unnormalised=pk)
        ctr = vr.contrast.reindex(g.index)
        for pn, pv in preds.items():
            within = [uq._spearman(pv[ctr == ct], g[ctr == ct]) for ct in ctr.unique()]
            rows.append(dict(R=R, predictor=pn, rho_all=uq._spearman(pv, g), rho_within_contrast=np.nanmean(within),
                             gain_cv=float(g.std() / max(abs(g.mean()), 1e-12))))
    tabC2 = pd.DataFrame(rows); display(tabC2.round(3))
    if 'R2_nex2' in CC and 'R2_nex1' in CC:
        g = (CC['R2_nex1'] - CC['R2_nex2']).dropna(); o = one.xs(2, level='R_nom').reindex(g.index)
        ac = uq.allocation_curve(CC.loc[g.index, 'R2_nex1'], g, o.voi_nex, cost['R2_nex1'], cost['R2_nex2'] - cost['R2_nex1'])
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for pol, d in ac.groupby('policy'): ax.plot(d.cost, d.err, '.-', label=pol)
        ax.set_xlabel('mean scan time'); ax.set_ylabel('mean NMSE'); ax.legend(fontsize=7)
        ax.set_title('R=2: give a 2nd average to the top-q slices (by analytic VOI)')
        plt.tight_layout(); plt.savefig(f'{OUT}/fig_C_allocation.png', dpi=130); plt.show()
""")

code(r"""
# D  don't trust: shift detection vs failure detection, and does an in-distribution threshold survive shift?
ID = A[(A.dataset == 'fastmri') & (A.R == 4)].assign(set='3T ID')
DD = res[res.probe == 'D'].assign(set=lambda d: '3T + ' + d.tag)
MM = A[(A.dataset == 'm4raw') & (A.R == 4)].assign(set=lambda d: '0.3T ' + d.contrast)
PD = pd.concat([ID, DD, MM], ignore_index=True)
units_id = sorted(ID.unit.unique()); fit_u = set(units_id[::2])
fit = PD[(PD.set == '3T ID') & PD.unit.isin(fit_u)]
for col in ('typ_xbar', 'typ_ref'):
    mu, Ci = uq.maha_fit(np.stack(fit[col].to_numpy()))
    PD[col + '_maha'] = uq.maha(np.stack(PD[col].to_numpy()), mu, Ci)
PD['inv_snr'] = -PD.snr_data
SC = {'pred_nmse': 'ensemble spread', 'cv_nmse': 'k-space hold-out residual', 'typ_xbar_maha': 'prior typicality (recon)',
      'inv_snr': 'SNR only (trivial)', 'typ_ref_maha': 'prior typicality (fully sampled; oracle)'}
DEPLOYABLE = ['pred_nmse', 'cv_nmse', 'typ_xbar_maha', 'inv_snr']
ev = PD[~((PD.set == '3T ID') & PD.unit.isin(fit_u))]
tau_D = float(ID.nmse.quantile(0.8))
ev = ev.assign(fail=ev.nmse > tau_D)
sets = [s for s in ev.set.unique() if s != '3T ID']
rows = []
for s, nm_ in SC.items():
    r = dict(score=nm_, failure_auroc_pooled=uq.auroc(ev[s], ev.fail))
    for st in sets:
        dd = ev[ev.set.isin(['3T ID', st])]
        r[f'shift AUROC {st}'] = uq.auroc(dd[s], dd.set == st)
        r[f'rho(score, NMSE) {st}'] = uq._spearman(ev[ev.set == st][s], ev[ev.set == st].nmse)
    rows.append(r)
tabD = pd.DataFrame(rows).set_index('score'); display(tabD.T.round(3))
print(f'failure = NMSE > {tau_D:.4g} (80th percentile of 3T in-distribution at R=4); '
      f'failure rate per set:', ev.groupby('set').fail.mean().round(2).to_dict())
# monitor validity: accept if score <= theta, theta set on the ID fit volumes so accepted violation <= alpha
rows = []
for s in DEPLOYABLE:
    fs = fit.assign(**{s: PD.loc[fit.index, s]}).sort_values(s)
    viol = (fs.nmse > tau_D).cumsum().to_numpy() / np.arange(1, len(fs) + 1)
    okk = np.flatnonzero(viol <= ALPHA)
    theta = fs[s].iloc[okk.max()] if len(okk) else -np.inf
    for st, d in ev.groupby('set'):
        acc = d[s] <= theta
        rows.append(dict(score=SC[s], set=st, accept_rate=acc.mean(),
                         viol_among_accepted=float((d[acc].nmse > tau_D).mean()) if acc.any() else np.nan))
tabD2 = pd.DataFrame(rows).pivot(index='set', columns='score', values=['accept_rate', 'viol_among_accepted'])
display(tabD2.round(3))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for j, s in enumerate(['pred_nmse', 'cv_nmse', 'typ_xbar_maha']):
    for st, d in ev.groupby('set'):
        ax[j].loglog(np.maximum(d[s], 1e-6), d.nmse, 'o', ms=3, alpha=.7, label=st)
    ax[j].axhline(tau_D, color='k', lw=.6); ax[j].set_xlabel(SC[s]); ax[j].set_ylabel('NMSE')
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig(f'{OUT}/fig_D_monitors.png', dpi=130); plt.show()
""")

code(r"""
# E  small findings at R = 8: kept or lost, and does uncertainty notice?
E = res[(res.probe == 'E') & (res.tag == 'metrics')]
LE = pd.DataFrame([dict(l, unit=r.unit, key=r.key) for _, r in E.iterrows() for l in r.lesions]) if len(E) else pd.DataFrame()
tabE = None
if len(LE):
    LE['lost'] = LE.cr < 0.5
    feats = {'unc_ratio_vs_clean': 'posterior var with / without lesion', 'unc_ratio_vs_ring': 'lesion var / surround var',
             'cr_sample_std': 'sample disagreement on the lesion', 'z_core': 'calibration ratio in lesion (needs truth)'}
    tabE = pd.DataFrame([dict(feature=v, auroc_lost=uq.auroc(LE[k], LE.lost), median=LE[k].median()) for k, v in feats.items()])
    print(f'{len(LE)} lesions: contrast recovery median {LE.cr.median():.2f}, lost (CR < 0.5): {LE.lost.mean():.0%}, '
          f'calibration ratio in lesion core {LE.z_core.median():.2f} vs surround {LE.z_ring.median():.2f}')
    display(tabE.round(3))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    sc_ = ax[0].scatter(LE.radius, LE.cr, c=LE.rel_contrast, s=14, cmap='viridis'); plt.colorbar(sc_, ax=ax[0], label='relative contrast')
    ax[0].axhline(0.5, color='k', lw=.6); ax[0].set_xlabel('radius (px)'); ax[0].set_ylabel('contrast recovery')
    ax[1].scatter(LE.unc_ratio_vs_clean, LE.cr, s=14); ax[1].set_xscale('log'); ax[1].axhline(0.5, color='k', lw=.6)
    ax[1].set_xlabel('posterior variance in lesion: with / without'); ax[1].set_ylabel('contrast recovery')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig_E_lesions.png', dpi=130); plt.show()
    exf = sorted(f for f in os.listdir(MAPS) if f.startswith('E_example')) if os.path.isdir(MAPS) else []
    if exf:
        z = np.load(f'{MAPS}/{exf[0]}'); cy, cx = z['centers'][0]; w = 24
        crop = (slice(max(cy - w, 0), cy + w), slice(max(cx - w, 0), cx + w))
        fig, ax = plt.subplots(1, 3, figsize=(8, 2.8))
        for j, (im, t) in enumerate([(z['ref'] + z['blob'], 'truth with lesion'), (z['les'], 'R=8 posterior mean'),
                                     (z['cln'], 'R=8 without lesion')]):
            ax[j].imshow(np.abs(im)[crop], cmap='gray'); ax[j].set_title(t, fontsize=8); ax[j].axis('off')
        plt.tight_layout(); plt.show()
""")

code(r"""
# F  FA-KGD as uncertainty: spectral vs isotropic DC (paired, R = 8), and the free analytic variance
FF = res[res.probe == 'F'].set_index('key')
iso = A[(A.dataset == 'fastmri') & (A.R == 8)].set_index('key').reindex(FF.index)
tabF = None
if len(FF):
    lr = lambda d: np.abs(np.log10((d.mse / d.pred_mse).clip(lower=1e-9)))
    tabF = pd.DataFrame([dict(dc=n, n=len(d), psnr_med=d.psnr.median(), calib_ratio=(d.mse / d.pred_mse).median(),
                              abs_log_calib=lr(d).median(), rho_ens=uq._spearman(d.pred_nmse, d.nmse),
                              patch_auc_pred=d.patch_auc_pred.median())
                         for n, d in (('isotropic', iso), ('spectral (FA-KGD)', FF))]).set_index('dc')
    tabF.loc['spectral - isotropic (paired dB)', 'psnr_med'] = float(np.median(10 * np.log10(iso.mse.clip(lower=1e-12) / FF.mse.clip(lower=1e-12))))
    display(tabF.round(3))
    cf_ = pd.DataFrame([dict(dc=n, **uq.column_stats(r)) for n, d in (('iso', iso), ('spectral', FF)) for _, r in d.iterrows()])
    display(cf_.groupby('dc').median().round(3))
print('free analytic UQ (F2): Spearman(analytic, NMSE) / Spearman(ensemble, NMSE) per group:')
display((tabA.rho_analytic / tabA.rho_ens).round(2))
""")

code(r"""
# G  where to measure: max-variance vs random vs nested next lines (same number of added lines, K = 4)
G = res[res.probe == 'G']
tabG = None
if len(G):
    nest = A[(A.dataset == 'fastmri') & (A.R == 4)].set_index('key')[f'mse_k{KS}']
    gp = G.pivot_table(index='key', columns='tag', values=f'mse_k{KS}').join(nest.rename('nested'), how='inner')
    tabG = pd.DataFrame({t: dict(median_db_vs_nested=float(np.median(10 * np.log10(gp.nested / gp[t]))),
                                 win_rate=float((gp[t] < gp.nested).mean())) for t in ('var', 'rand') if t in gp}).T
    display(tabG.round(3))
""")

code(r"""
# H  (optional) post-trained 0.3T prior vs zero-shot, R = 4
HH = res[res.probe == 'H'].set_index('key')
tabH = None
if len(HH):
    z0 = A[(A.dataset == 'm4raw') & (A.R == 4)].set_index('key').reindex(HH.index)
    e_ = lambda d: (d[f'cr_k{K}'] - d.v_rep).clip(lower=1e-12)
    tabH = pd.DataFrame([dict(prior=n, nmse_med=(e_(d) / d.sig_pow).median(), calib_ratio=(d.mse / d.pred_mse).median(),
                              rho_ens=uq._spearman(d.pred_nmse, d.nmse)) for n, d in (('zero-shot', z0), ('post-trained', HH))])
    tabH['db_gain_vs_zero_shot'] = [0.0, float(np.median(10 * np.log10(e_(z0) / e_(HH))))]
    display(tabH.round(3))
""")

# =============================================================================
md(r"""
## Scoreboard
Metric → verdict against the pre-set criteria. Next to each row is the story it would support.
- **B + A1 promising:** a diffusion-posterior stopping rule for nested re-acquisition, with a
  conformal / Learn-then-Test guarantee at the stopping time. This is Wen et al. with a diffusion prior and a
  calibrated stop: an optimal-stopping / chance-constrained story, a SISL fit.
- **C1/C2:** "measure longer or measure more?", with uncertainty-driven averaging at low field. It
  appears to be open, and it extends the MIDL repetition project.
- **D:** runtime monitors for diffusion reconstruction. Hold-out residual vs posterior spread under shift.
- **E (lost lesions, and uncertainty does not notice):** a strong cautionary finding about
  "hallucinated normality".
- **F:** FA-KGD repositioned as free, calibrated uncertainty.
""")

code(r"""
# scoreboard
def g(df, idx, col):
    try:
        v = df.loc[idx, col]
        return float(v) if np.isfinite(v) else np.nan
    except Exception:  # noqa: BLE001
        return np.nan

main_groups = [x for x in ('fastmri R8', 'fastmri R4', 'm4raw R4', 'm4raw R2') if x in tabA.index]
val = {}
val['A1'] = float(np.nanmedian([tabA.loc[x, 'rho_ens'] for x in main_groups]))
val['A2'] = float(np.nanmedian([tabA.loc[x, 'patch_auc_pred'] - tabA.loc[x, 'patch_auc_edge'] for x in main_groups]))
val['A3'] = float(np.nanmax(np.abs(tabA3.coverage_at_transferred_thr - 0.9)))
sel = lambda tb: tb[(tb.viol_target == 0.10) & (tb.tau == 'primary')].set_index('score').captured
val['B'] = g(sel(tbB_f).to_frame(), SCORES['pred'], 'captured')
val['B_m4'] = g(sel(tbB_m).to_frame(), SCORES['pred'], 'captured') if tbB_m is not None else np.nan
val['B_crc'] = float((ltB_f['crc'].test_viol > ALPHA + 0.05).mean())
val['C1'] = float(tabC1[tabC1.baseline.str.startswith('R')].win_rate.max()) if CC is not None and len(tabC1) else np.nan
val['C2'] = float(tabC2.rho_within_contrast.max()) if CC is not None and len(tabC2) else np.nan
val['D1'] = float(tabD.loc[[SC[s] for s in DEPLOYABLE], 'failure_auroc_pooled'].max())
v2 = tabD2['viol_among_accepted'].drop(index='3T ID', errors='ignore')
best_d = v2.max().idxmin() if len(v2) else None
val['D2'] = float(v2[best_d].max() / ALPHA) if best_d else np.nan
val['E'] = float(tabE.iloc[:3].auroc_lost.max()) if tabE is not None else np.nan
val['F1'] = float(tabF.loc['isotropic', 'abs_log_calib'] - tabF.loc['spectral (FA-KGD)', 'abs_log_calib']) if tabF is not None else np.nan
val['F2'] = float(np.nanmedian([(tabA.rho_analytic / tabA.rho_ens)[x] for x in main_groups]))
val['G'] = g(tabG, 'var', 'median_db_vs_nested') if tabG is not None else np.nan
val['H'] = float(tabH.db_gain_vs_zero_shot.iloc[1]) if tabH is not None else np.nan
board = pd.DataFrame([dict(id=k, criterion=c[0], value=val.get(k, np.nan), promising=c[1], weak=c[2],
                           verdict=uq.verdict(val.get(k, np.nan), c[1], c[2], c[3])) for k, c in CRITERIA.items()]).set_index('id')
pd.set_option('display.max_colwidth', 90)
display(board.round(3))
notes = []
if tabE is not None and LE.lost.mean() >= 0.2 and val['E'] < 0.6:
    notes.append('E: >= 20% of small lesions are lost at R=8 and uncertainty does not flag them (silent loss).')
if np.isfinite(val['A1']) and tabA.calib_ratio.median() > 2:
    notes.append(f'A: ensembles are overconfident (median calibration ratio {tabA.calib_ratio.median():.1f}); '
                 'decisions need conformal recalibration (B uses LTT for this).')
if best_d:
    notes.append(f'D2: best monitor = {best_d}.')
print('\n'.join(notes))
board.to_csv(f'{OUT}/scoreboard.csv')
json.dump(dict(values=val, T=T, K=K, n_fastmri=len(CF), n_m4raw=len(CM), minutes=(time.time() - T_START) / 60,
               tables={k: v.reset_index().to_dict(orient='records') for k, v in
                       dict(A=tabA, A3=tabA3).items()}), open(f'{OUT}/summary.json', 'w'), indent=1, default=str)
sync(verbose=True)
print(f'total wall time {(time.time() - T_START) / 60:.1f} min')
""")

md(r"""
## Key references (checked 2026-10-13)
- **Conformal multi-round MRI.** Wen, Ahmad, Schniter. Task-driven UQ in inverse problems via conformal prediction. ECCV 2024. https://arxiv.org/abs/2405.18527. Also their conformal bounds on full-reference image quality, TMLR 2025: https://arxiv.org/abs/2505.09528
- **Stopping rule losing coverage.** Fischer et al. CUTE-MRI, 2025. Coverage at the stopping time falls below target. https://arxiv.org/abs/2508.14952
- **Calibration tools.** Conformal risk control, ICLR 2024: https://arxiv.org/abs/2208.02814. Learn-then-Test, Ann. Appl. Stat. 2025: https://doi.org/10.1214/24-AOAS1998
- **Image-to-image conformal and diffusion intervals.** Angelopoulos et al., ICML 2022: https://arxiv.org/abs/2202.05265. Teneggi et al., K-RCPS, ICML 2023: https://arxiv.org/abs/2302.03791
- **Diffusion posterior variance is not an error map:**
  - Qiu et al. 2026: https://arxiv.org/abs/2602.04189
  - Zhao et al. 2026: https://arxiv.org/abs/2607.13682
  - Thong, Mbakam & Pereyra 2024: https://arxiv.org/abs/2405.08179
- **Posterior-variance error bounds.** Narnhofer et al., SIIMS 2024. https://arxiv.org/abs/2212.12499
- **Adaptive sampling.**
  - Zhang et al., CVPR 2019: https://arxiv.org/abs/1902.03051
  - Bakker et al., MIDL 2022 (non-adaptive is often enough): https://arxiv.org/abs/2203.16392
  - AdaSense, ECCV 2024: https://arxiv.org/abs/2407.08256
  - Iollo et al. 2025: https://arxiv.org/abs/2506.16237
- **Low-field averaging.**
  - Schoormans et al., PMB 2020 (variable-density averaging): https://arxiv.org/abs/1909.01672
  - NexOP, 2026: https://arxiv.org/abs/2605.11583
  - M4Raw: https://doi.org/10.1038/s41597-023-02181-4
- **OOD with score and diffusion models.**
  - Mahmood et al., ICLR 2021: https://arxiv.org/abs/2010.13132
  - Graham et al., CVPR-W 2023: https://arxiv.org/abs/2211.07740
  - DiffPath, NeurIPS 2024: https://arxiv.org/abs/2405.11881
- **Hold-out validation of a reconstruction.**
  - Ward, IEEE TIT 2009: https://arxiv.org/abs/0803.1845
  - SSDU, MRM 2020: https://arxiv.org/abs/1912.07669
- **Hallucinations.** Bhadra et al., TMI 2021: https://arxiv.org/abs/2012.00646. fastMRI 2020 challenge, TMI 2021: https://arxiv.org/abs/2012.06318
- **SISL.**
  - Moss et al., ConstrainedZero, IJCAI 2024 (conformal thresholds in chance-constrained POMDPs): https://arxiv.org/abs/2405.00644
  - Betterton et al., ICRA 2020 (adaptive X-ray acquisition time): https://doi.org/10.1109/ICRA40945.2020.9196614
  - Corso et al., JAIR 2021 (black-box safety validation): https://arxiv.org/abs/2005.02979
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = HERE / "uq_pilot_colab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
