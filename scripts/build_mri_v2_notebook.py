"""(Re)generate notebooks/colab_pvas_mri_v2.ipynb — full-scale MRI sweep for the
WACV paper (192-slice gate-off production numbers, both sigma_max regimes,
ensemble head-to-heads, knee OOD, multi-seed masks, author-code adapters).

Run:  python scripts/build_mri_v2_notebook.py
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colab_pvas_mri_v2.ipynb"


_ids = iter(range(1000))


def md(s):
    return {"cell_type": "markdown", "id": f"md-{next(_ids)}", "metadata": {},
            "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "id": f"code-{next(_ids)}", "metadata": {},
            "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


# Gate-off selection-only method set (production v2): scalar PiGDM gain
# everywhere, PV computed only at the acquisition rounds. NFE accounting per
# slice at T=20, q=pv_n_probes=2, R_a=active_rounds=2:
#   pv_active (gate-off):        T + 2*q*R_a = 28   (1.4x passive)
#   ens_active / ens_random:     N*T          = 80  at N=4 (4x passive)

cells = [
    md("""# PVAS — MRI domain, production sweep v2 (WACV paper)

Gate-off **selection-only** PVAS at 192-slice scale: PV is computed only at the
2 acquisition rounds (q=2 Hutchinson probes each), scalar PiGDM gain
everywhere, so NFE = T + 2qR_a = 20 + 8 = **28** per slice (1.4x the passive
PiGDM baseline). The ADS-style particle-ensemble baselines (`ens_active`,
`ens_random_active`, reimplemented in our pipeline: N parallel PiGDM particles,
cross-particle variance of predicted k-space columns, mean recon) run
head-to-head at matched budget with their NFE = N*T reported.

Sections:
- §A/§B — 192-slice (12-volume AXFLAIR, 3 chunks of 4) grids, R in {4,8} x
  L in {8,16}, sigma_max in {10, 80} -> `outputs/PVAS_full*/`.
- §C — ensemble particle sweep N in {2,4,8,16}.
- §D — knee OOD (brain prior on knee data).
- §E — multi-seed mask realizations (mask_seed 42/43/44).
- §F — author-code head-to-head adapters (ADS / AdaSense / Iollo et al.).
- §G — summarize into paper tables + copy back to Drive.

Needs: A100 recommended; `fastmri_artifacts` on Drive with the EDM checkpoint
and 12 brain multicoil AXFLAIR validation volumes (same layout as
`colab_pvas_sweep.ipynb`)."""),
    code("!nvidia-smi | head -5"),
    code("""# Cell 2 — clone repo + ADPS (EDM pickle deps)
import os
REPO='/content/fastmri'
if not os.path.exists(REPO):
    !git clone --depth 1 https://github.com/carlo-scr/fastmri.git {REPO}
    !git clone --depth 1 https://github.com/utcsilab/ambient-diffusion-mri.git {REPO}/external/adps
%cd {REPO}
!pip install -q h5py scikit-image scipy pyyaml"""),
    code("""# Cell 3 — mount Drive, stage checkpoint + data, build the 3 volume chunks
from google.colab import drive; drive.mount('/content/drive')
import os, shutil, glob
ART='/content/drive/MyDrive/fastmri_artifacts'
os.makedirs('checkpoints/edm/supervised_R=1', exist_ok=True)
os.makedirs('data/multicoil_val', exist_ok=True)
if not os.path.exists('checkpoints/edm/supervised_R=1/network-snapshot.pkl'):
    shutil.copy(f'{ART}/network-snapshot.pkl', 'checkpoints/edm/supervised_R=1/')
    shutil.copy(f'{ART}/training_options.json', 'checkpoints/edm/supervised_R=1/')
if not glob.glob('data/multicoil_val/*.h5'):
    for f in glob.glob(f'{ART}/multicoil_val/*.h5'):
        shutil.copy(f, 'data/multicoil_val/')
vols = sorted(glob.glob('data/multicoil_val/*.h5'))
print(len(vols), 'volumes staged')
assert len(vols) >= 12, 'need 12 AXFLAIR volumes for the 3-chunk production grid'
# 3 chunks of 4 volumes each (matches scripts/run_production_v2.sh layout) so
# partial grids still yield usable results.json files per chunk.
for c in range(3):
    d = f'data/axflair_chunk{c+1}'
    os.makedirs(d, exist_ok=True)
    for f in vols[4*c:4*(c+1)]:
        dst = os.path.join(d, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(f), dst)
    print(d, '->', len(glob.glob(d + '/*.h5')), 'volumes')"""),
    md("""## §A — 192-slice gate-off grid, sigma_max=10 (tab:mri_main)

Headline configs (R8_a16, R4_a8) run first; per-chunk `results.json` files are
merged by `gen_tables.py`/analysis scripts. Each config x chunk is
independently resumable (skipped if its results.json exists)."""),
    code("""# Cell 4 — R x L grid, gate-off + faithful ensemble baselines, 3 chunks
import os
for R, L in [(8,16), (4,8), (4,16), (8,8)]:
    for c in [1, 2, 3]:
        out = f'outputs/PVAS_full/R{R}_a{L}_chunk{c}'
        if os.path.exists(f'{out}/results.json'): print('skip', out); continue
        !python scripts/reconstruct_mc.py --data_path data/axflair_chunk{c} \\
          --checkpoint_dir checkpoints/edm/supervised_R=1 \\
          --num_volumes 4 --num_slices 0 --num_steps 20 --sigma_max 10 \\
          --target_resolution 384 320 --device cuda \\
          --acceleration {R} --active_lines {L} \\
          --active_rounds 2 --active_gate off --ens_particles 4 \\
          --methods pigdm_mc pv_active random_adaptive static_match oracle_active ens_active ens_random_active \\
          --output_dir {out}"""),
    md("""## §B — 192-slice grid, sigma_max=80 regime (tab:mri80)

Headline configs (R4_a8, R8_a16) first; (4,16)/(8,8) are supplementary — stop
after the first two config loops if GPU time is short."""),
    code("""# Cell 5 — sigma_max=80, same method set
import os
for R, L in [(4,8), (8,16), (4,16), (8,8)]:
    for c in [1, 2, 3]:
        out = f'outputs/PVAS_full80/R{R}_a{L}_chunk{c}'
        if os.path.exists(f'{out}/results.json'): print('skip', out); continue
        !python scripts/reconstruct_mc.py --data_path data/axflair_chunk{c} \\
          --checkpoint_dir checkpoints/edm/supervised_R=1 \\
          --num_volumes 4 --num_slices 0 --num_steps 20 --sigma_max 80 \\
          --target_resolution 384 320 --device cuda \\
          --acceleration {R} --active_lines {L} \\
          --active_rounds 2 --active_gate off --ens_particles 4 \\
          --methods pigdm_mc pv_active random_adaptive static_match oracle_active ens_active ens_random_active \\
          --output_dir {out}"""),
    md("""## §C — ensemble particle sweep (N in {2, 4, 8, 16})

Cost/quality spectrum for the faithful ADS-style baseline on the headline
config (R=8, L=16, sigma_max=10): NFE = N*T = {40, 80, 160, 320} vs
pv_active's 28. 36 slices (12 volumes x 3) keep N=16 affordable; pv_active is
re-run in the same command so every comparison is paired on identical
slices/masks."""),
    code("""# Cell 6 — particle sweep at matched budget
import os
for N in [2, 4, 8, 16]:
    out = f'outputs/PVAS_ens_sweep/R8_a16_N{N}'
    if os.path.exists(f'{out}/results.json'): print('skip', out); continue
    !python scripts/reconstruct_mc.py --data_path data/multicoil_val \\
      --checkpoint_dir checkpoints/edm/supervised_R=1 \\
      --num_volumes 12 --num_slices 3 --num_steps 20 --sigma_max 10 \\
      --target_resolution 384 320 --device cuda \\
      --acceleration 8 --active_lines 16 \\
      --active_rounds 2 --active_gate off --ens_particles {N} \\
      --methods pigdm_mc pv_active ens_active ens_random_active \\
      --output_dir {out}"""),
    md("""## §D — knee OOD (brain prior on knee data)

Out-of-distribution stress test: the brain-trained EDM prior reconstructs
fastMRI **multicoil knee** volumes; adaptivity should matter more when the
prior is mismatched. Data note: `scripts/reconstruct_mc.py` requires MULTICOIL
h5 volumes. The legacy singlecoil knee path
(`data/singlecoil_test/file1000022.h5`, used by `scripts/run_adps_experiments.py`
and the `outputs/edm_knee_*` runs) is NOT compatible with this multicoil
pipeline — it exists only for the older single-coil comparisons.
TODO(data): place >= 4 knee multicoil val volumes (CORPD) in Drive under
`fastmri_artifacts/knee_multicoil_val/` (download: `scripts/download_data.sh`,
option 5)."""),
    code("""# Cell 7 — stage knee volumes + run headline configs
import glob, os, shutil
os.makedirs('data/knee_multicoil_val', exist_ok=True)
if not glob.glob('data/knee_multicoil_val/*.h5'):
    for f in sorted(glob.glob('/content/drive/MyDrive/fastmri_artifacts/knee_multicoil_val/*.h5'))[:4]:
        shutil.copy(f, 'data/knee_multicoil_val/')
n_knee = len(glob.glob('data/knee_multicoil_val/*.h5'))
print(n_knee, 'knee volumes staged')
if n_knee == 0:
    print('SKIPPING knee OOD — stage volumes first (see markdown above)')
else:
    for R, L in [(4,8), (8,16)]:
        out = f'outputs/PVAS_knee_ood/R{R}_a{L}'
        if os.path.exists(f'{out}/results.json'): print('skip', out); continue
        !python scripts/reconstruct_mc.py --data_path data/knee_multicoil_val \\
          --checkpoint_dir checkpoints/edm/supervised_R=1 \\
          --num_volumes 4 --num_slices 0 --num_steps 20 --sigma_max 10 \\
          --target_resolution 384 320 --device cuda \\
          --acceleration {R} --active_lines {L} \\
          --active_rounds 2 --active_gate off --ens_particles 4 \\
          --methods pigdm_mc pv_active random_adaptive static_match oracle_active ens_active ens_random_active \\
          --output_dir {out}"""),
    md("""## §E — multi-seed mask realizations (mask_seed 42 / 43 / 44)

Robustness of the selection gain to the random undersampling mask: same
headline config (R=8, L=16, sigma_max=10), three mask seeds, 36 slices each.
mask_seed=42 replicates the main-grid mask; 43/44 are fresh realizations."""),
    code("""# Cell 8 — mask-seed sweep
import os
for ms in [42, 43, 44]:
    out = f'outputs/PVAS_maskseed/R8_a16_ms{ms}'
    if os.path.exists(f'{out}/results.json'): print('skip', out); continue
    !python scripts/reconstruct_mc.py --data_path data/multicoil_val \\
      --checkpoint_dir checkpoints/edm/supervised_R=1 \\
      --num_volumes 12 --num_slices 3 --num_steps 20 --sigma_max 10 \\
      --target_resolution 384 320 --device cuda --mask_seed {ms} \\
      --acceleration 8 --active_lines 16 \\
      --active_rounds 2 --active_gate off --ens_particles 4 \\
      --methods pigdm_mc pv_active random_adaptive static_match \\
      --output_dir {out}"""),
    md("""## §F — author-code head-to-head (ADS / AdaSense / Iollo et al.)

Our pipeline already runs a **faithful reimplementation** of the ADS-style
particle-ensemble baseline (`ens_active` above: N parallel PiGDM particles,
cross-particle variance of predicted k-space columns, mean recon) at matched
budget AND matched prior. The cells below additionally prepare runs of the
authors' own code; honest TODO markers flag every place where their configs
must be chosen by hand.

### Config mapping — ADS (Nolan et al., TMLR 2025)

| knob | ADS (author code) | PVAS (ours) |
|---|---|---|
| uncertainty | Np=16 parallel DPS particles, GMM entropy of particle Tweedie means in measurement space | 1 trajectory, Tweedie PV at 2 acquisition rounds (q=2 probes) |
| budget | k-space columns acquired on their schedule | L in {8,16} active lines on top of 0.08 center fraction |
| solver / prior | DPS + their fastMRI score model | PiGDM scalar gain + our EDM brain multicoil checkpoint |
| NFE/slice | Np x T | T + 2qR_a = 28 |
| metric | PSNR/SSIM on per-slice-normalized magnitude | SENSE-referenced PSNR/SSIM, 320 center crop |

Two arms: (1) their code + their prior (author-faithful), (2) their selection
rule inside our pipeline (= `ens_active`, already run above). Metric
conversion: recompute BOTH conventions from saved reconstructions — never
compare numbers across conventions.

### Config mapping — AdaSense (Elata et al., ECCV 2024, code: github.com/noamelata/AdaSense)

| knob | AdaSense (author code) | PVAS (ours) |
|---|---|---|
| uncertainty | s complete DDRM posterior-sampling runs per acquisition round (s up to 256) | 1 trajectory, PV at 2 rounds |
| selection | top eigenvector / projected posterior variance of the sample covariance, between complete runs | per-column PV score, mid-trajectory |
| budget | measurement rows (general CS) or k-space lines (their MRI config) | L active lines |
| noise model | noiseless, hard DDRM consistency | noisy, scalar PiGDM gain |
| NFE/image | s x T x rounds | T + 2qR_a |

### Iollo et al. (BED-MRI, arXiv:2506.16237)

Training-free sequential BED/EIG with a diffusion prior; needs an N+M ensemble
of trajectories and selects between complete runs; benchmarks against ADS.
TODO(code): no public repo verified as of 2026-07-08 — check the arXiv page /
HAL for a code link before submission; until then this comparison stays
[CITED]-only in the paper."""),
    code("""# Cell 9 — ADS author-code adapter
!git clone https://github.com/active-diffusion-subsampling/ads /content/ads || true
%cd /content/ads
!pip install -q -r requirements.txt || echo 'TODO: resolve ADS deps by hand'
%cd /content/fastmri
# TODO(config): pick the ADS fastMRI config closest to our protocol —
#   acceleration R=8 total columns, ACS fraction 0.08, Np=16 particles (their
#   default), T matched to their paper (NOT our T=20; report both NFE columns).
# TODO(data): export our 12 AXFLAIR volumes to the ADS expected input format
#   (their dataloader expects single-coil/RSS magnitude — document the
#   conversion, it changes the task; run their knee config as-is for the
#   author-faithful arm instead if conversion distorts the comparison).
# Template (fix paths/config after reading /content/ads/README):
# !python /content/ads/main.py --config <their_fastmri_config>.yaml \\
#   --budget <match L=16> --particles 16 --output_dir outputs/ads_author_run
# Metric conversion (run after their run completes):
# TODO(metrics): load their saved recons + our SENSE references; report
#   (a) their magnitude-PSNR convention on both methods,
#   (b) our SENSE-PSNR only where a complex/coil-resolved recon exists;
#   never mix conventions in one table row.
print('ADS adapter prepared — resolve the TODO(config)/TODO(data) markers above')"""),
    code("""# Cell 10 — AdaSense author-code adapter
!git clone https://github.com/noamelata/AdaSense /content/adasense || true
%cd /content/adasense
!pip install -q -r requirements.txt || echo 'TODO: resolve AdaSense deps by hand'
%cd /content/fastmri
# TODO(config): choose their MRI line-acquisition entry point and set
#   s (posterior runs per round) and the number of rounds so the total line
#   budget equals ours (ACS + L). Their faces demo uses s=256 full DDRM runs
#   per round — record their actual s from the config you pick.
# TODO(prior): author-faithful arm uses their checkpoint; matched-prior arm
#   would need a DDRM-compatible wrapper of our EDM model (document if skipped).
# Template (fix after reading /content/adasense/README):
# !python /content/adasense/<their_mri_script>.py --num_samples <s> \\
#   --budget <match ACS+L> --output_dir outputs/adasense_author_run
# TODO(metrics): same two-convention reporting rule as the ADS cell.
print('AdaSense adapter prepared — resolve the TODO markers above')"""),
    code("""# Cell 11 — Iollo et al. (BED-MRI) adapter
# TODO(code): no public repository verified as of 2026-07-08. Check:
#   https://arxiv.org/abs/2506.16237 (landing page + comments) for a code link.
# If code appears: map their N+M trajectory ensemble and EIG line scoring to
#   our budget (ACS 0.08 + L lines), run on the same 12 AXFLAIR volumes, and
#   report their NFE = (N+M)*T next to ours (28).
# Until then the paper cites their reported numbers as [CITED] only.
print('Iollo adapter: TODO(code) — no verified public repo yet')"""),
    md("## §G — summarize into paper tables + copy back to Drive"),
    code("""# Cell 12 — regenerate paper tables and print grid summaries
!python paper/pvas_v2/tables/gen_tables.py || true
import json, glob
roots = ['outputs/PVAS_full*/**/results.json', 'outputs/PVAS_ens_sweep/**/results.json',
         'outputs/PVAS_knee_ood/**/results.json', 'outputs/PVAS_maskseed/**/results.json']
for pat in roots:
    for p in sorted(glob.glob(pat, recursive=True)):
        r = json.load(open(p)); s = r.get('summary', {})
        print(p)
        for k, v in s.items():
            if isinstance(v, dict) and 'psnr_sense_mean' in v:
                print(f"   {k:22s} {v['psnr_sense_mean']:.3f}")"""),
    code("""# Cell 13 — copy results back to Drive
!mkdir -p /content/drive/MyDrive/fastmri_artifacts/results_v2
!cp -r outputs/PVAS_full outputs/PVAS_full80 outputs/PVAS_ens_sweep \\
       outputs/PVAS_knee_ood outputs/PVAS_maskseed \\
       /content/drive/MyDrive/fastmri_artifacts/results_v2/ 2>/dev/null || true
!ls /content/drive/MyDrive/fastmri_artifacts/results_v2/"""),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
NB.write_text(json.dumps(nb, indent=1))
print("wrote", NB)
