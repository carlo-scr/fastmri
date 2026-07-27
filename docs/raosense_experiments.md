# RaoSense WACV experiment preregistration

**Protocol version:** 0.1

**Frozen inference seeds:** 2027, 2028, 2029

**Intended venue:** WACV main conference

**Status:** design and synthetic falsification only; all real MRI and LiDAR
results are **PENDING**

This document is the experiment contract for testing the central RaoSense
claim. It is deliberately stricter than a paper outline: it fixes the
datasets, independent units, domain-specific observation budgets, compute accounting,
comparators, endpoints, and decision gates before the locked test sets are
opened.

The intended claim is:

> An operator-sufficient, Rao--Blackwellized mixture belief can select useful
> measurements during diffusion posterior sampling more accurately and more
> efficiently than point-particle uncertainty, and outcome-contingent
> two-step planning provides an additional benefit. The same inference
> algorithm should work for global complex-valued parallel MRI and local
> coordinate-selection LiDAR sensing; only the forward operator and
> emulator-legal action set change.

The experiments do **not** claim prospective scanner or programmable-LiDAR
hardware validation. Both studies are retrospective sensing emulations in
which the complete measurement is hidden from the policy and revealed only
after a legal action is selected.

## 1. Evidence ledger

### 1.1 Completed evidence

The following checks are complete, but they are analytic or synthetic checks,
not imaging evidence.

| Evidence | Result | Interpretation |
|---|---:|---|
| Rao--Blackwell covariance simulation | particle covariance MSE 0.374568; Rao--Blackwell MSE 0.143969; 61.5639% reduction | The estimator can reduce covariance error under its assumed mixture model. |
| Operator-sufficient compression | measured rank 11 for 4,096 scalar measurements; 372.36x compression | The synthetic belief occupied the predicted low-dimensional subspace. |
| Information sandwich | estimate 4.157348, SE 0.003941, inside [2.869404, 4.255699] | The estimator respected the analytic bounds in the synthetic case. |
| Outcome routing | one-step chose action 0; horizon two chose action 1 | The constructed example contains a real nonmyopic decision. |
| Unknown-operator check | target information 0.175699 when uncertainty is modeled versus 1.098612 when the operator is treated as known, a 6.2528x ratio | Ignoring forward-model uncertainty can be strongly optimistic. |
| Latest completed full CPU suite (2026-07-26) | 43 passed in 8.09 seconds at code-under-test commit `03746c938bd09e1bad620353d851ac32818d4ece` | Analytic, protocol-regression, MRI/LiDAR adapter, paired H=1/H=2 zero-future equivalence, budget, VP-transition, observation-separation, and nuisance tests pass; this is not GPU imaging evidence. |
| LiDAR hidden-truth separation | `observed_scan` is now the sole conditioning and data-consistency source, with a regression introduced at code commit `6f6bee8a319a971a6091628041b812f23466292a` | A once-sampled corrupted observation can be replayed without clamping hidden truth; the later full suite above passes with this change. |

The numerical artifact is
`outputs/raosense/analytic_validation.json`, generated at Git commit
`03746c938bd09e1bad620353d851ac32818d4ece` with seed 2027, 300 covariance
trials, 12 particles, 4,096 measurement dimensions, and 10,000 information
samples. The current full-suite command was:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

The protocol file itself is still untracked at the code-under-test commit.
Before S1, commit the protocol separately, record that protocol commit and
clean/dirty state, and retain the complete test log plus SHA256. These checks
must be rerun at the final evidence commit. Numbers from older frequency-domain
or PV experiments marked `LEGACY-INVALIDATED` elsewhere in the repository are
not evidence for RaoSense.

### 1.2 Evidence that is still PENDING

As of this protocol version:

- **PENDING:** no real fastMRI brain result;
- **PENDING:** no real fastMRI knee result;
- **PENDING:** no fastMRI+ pathology or hallucination evaluation;
- **PENDING:** no KITTI-360/R2DM reconstruction or acquisition result;
- **PENDING:** no matched wall-clock benchmark on A800 GPUs;
- **PENDING:** no learned external baseline has been ported to the common
  action/budget interface;
- **PENDING:** the adapter still defaults to a 48-candidate engineering
  prefilter, which is unsafe for H=2 because immediate information cannot
  upper-bound future value; the production driver must force an uncapped root
  set and assert `planner_candidate_cap is None`;
- **PENDING:** no production LiDAR command-line driver exists around
  `src/samplers/raosense_lidar.py`;
- **PENDING:** no real LiDAR corrupted-observation robustness run or
  calibration-fitted corruption artifact exists; `observed_scan` separation
  is implemented, but a categorical validity likelihood remains outside the
  method and must not be claimed;
- **PENDING:** `scripts/reconstruct_mc.py` computes image metrics and masks but
  does not yet retain the reconstructions needed for pathology-box, detector,
  or sFRC analysis.

No paper table may replace a PENDING cell with a toy-adapter number.

## 2. Frozen hypotheses and primary configuration

### 2.1 Hypotheses

The confirmatory hypotheses are:

1. **Belief hypothesis:** at the same domain-specific observation budget, total
   RaoSense uncertainty (between-mode plus within-mode) selects masks that
   outperform point-particle/ensemble variance and the endpoint-specific
   nonoracle comparators selected on calibration data.
2. **Planning hypothesis:** outcome-contingent horizon-two planning improves
   on the otherwise identical one-step RaoSense policy.
3. **Efficiency hypothesis:** the operator-sufficient belief is competitive
   with an in-family larger ensemble under exact-NFE and wall-clock
   comparisons, with NFE, total peak memory, incremental belief-state memory,
   and planner time reported separately.
4. **Generality hypothesis:** the gains hold for fastMRI+ brain, fastMRI+ knee,
   and KITTI-360 LiDAR despite the very different forward operators and action
   geometry.

### 2.2 Frozen headline configuration

The headline configuration is shared across domains:

| Symbol | Setting | Meaning |
|---|---:|---|
| \(K\) | 8 | measurement-guided diffusion branches |
| \(T\) | 32 | reverse steps |
| \(q\) | 2 | central finite-difference/JVP factors per branch and acquisition round |
| \(R\) | 4 | acquisition rounds |
| \(H\) | 2 | planning horizon |
| \(B\) | 8 | H=2 root beam width |
| \(F\) | 16 | fantasy outcomes per retained root |
| \(S\) | 128 | samples for mixture information estimation |
| \(C\) | all | no root prefilter in confirmatory H=2 planning |
| acquisition checkpoints | 9, 14, 18, 23 | zero-indexed denoiser steps from `build_active_schedule(32, ..., 4, 0.30, 0.75)` |
| finite-difference step | 0.05 | fraction of the current diffusion scale |

On calibration data, run every required comparator candidate with streams
2027, 2028, and 2029 before selecting an endpoint comparator. For a learned
policy, pair training seed 2027 with inference/replay seed 2027, and likewise
for 2028 and 2029: this is three total runs, not a 3x3 cross-product. For a
nonlearned policy, the three runs differ only in inference/replay randomness.
Select comparators from the three-run mean.

On locked data, run each selected endpoint comparator, the \(K=12\) ensemble,
RaoSense H=1, and RaoSense H=2 with those three paired streams and average
within study/frame before statistical resampling. Descriptive baselines whose
comparator status was ruled out by the complete three-stream calibration may
use stream 2027 only. The \(K=16\) representation reference is an explicit
single-stream exception and is compared only with RaoSense stream 2027 in
Gate 5. The per-stream RaoSense/K=12 online cost is 384 NFE; report both 384
per reconstruction and 1,152 aggregate research NFE for a three-stream
estimate.

The denoiser-call count per slice or frame is

\[
\mathrm{NFE}=KT+2KqR
=8(32)+2(8)(2)(4)
=384.
\]

H=1 and H=2 use the same 384 NFE. The H=2 planner performs additional
low-dimensional mixture calculations, so NFE equality does not imply
wall-clock equality. Every result must report both.

All method-specific tuning ends on the calibration split. The locked test
configuration is immutable. A deviation requires a new protocol version and a
new locked split; it cannot be described as a harmless rerun.

Analytic information bounds may safely screen actions only for a one-step
objective. They do not bound an H=2 root's outcome-contingent future value.
The 48-candidate cap is therefore an engineering ablation, not the headline
method. Confirmatory H=2 scores every legal root, then applies beam width 8 to
future rollout.

The paired planning intervention also requires common random numbers. At each
state, compute and cache one action-keyed immediate-information estimate for
every legal root. H=1 maximizes exactly that cache; H=2 uses the same cache and
adds only expected future value. Screening must not change which Monte Carlo
draws an action receives. A regression test must show that H=2 with its future
term forced to zero returns exactly the H=1 action and objective.

## 3. Data governance, provenance, and storage

### 3.1 fastMRI and fastMRI+

Use raw fully sampled, multicoil knee and brain acquisitions from the
[fastMRI dataset owner](https://fastmri.org/), loaded according to the
[official fastMRI repository](https://github.com/facebookresearch/fastMRI).
The official loader documents that one HDF5 file contains one MR acquisition.
The knee release and storage layout are described in the
[dataset paper](https://pubs.rsna.org/doi/10.1148/ryai.2020190007):
multicoil train is 931 GB, multicoil validation is 192 GB, and the complete
k-space release described there is approximately 1.35 TB.

Download raw data only through an authorized fastMRI account after accepting
the current data-sharing agreement. The raw files, signed URLs, account
credentials, and any access tokens must never enter Git, a public artifact, or
a paper supplement. The agreement, not this protocol, is authoritative. At
minimum:

- keep raw files in access-controlled storage;
- record the dataset snapshot/download date and SHA256 of every used HDF5
  file;
- do not redistribute raw measurements or derived per-case variables/data
  files unless NYU has explicitly permitted the planned release;
- use the data only within the permitted research scope;
- include “NYU fastMRI” in the abstract, the required dataset citations and
  source acknowledgement in Methods, and the current investigator disclaimer;
- contact the address on the portal for permission before publishing dataset
  images;
- maintain a project closeout date and destruction log for all dataset copies;
- remove any sample from the manifest if its provenance is ambiguous.

These obligations come from the
[current NYU fastMRI Dataset Sharing Agreement](https://fastmri.med.nyu.edu/),
which permits internal research/education, restricts distribution of both data
and derived data files, prescribes publication language, prohibits clinical
use, and requires destruction of copies when the work is complete. Gate 0
requires the actual accepted agreement to be reviewed again because the portal
states that it can be updated.

Clinical localization comes from
[fastMRI+](https://www.nature.com/articles/s41597-022-01255-z) and its
[official annotation repository](https://github.com/microsoft/fastmri-plus).
fastMRI+ contains 16,154 knee boxes and 13 study-level labels spanning 22
categories on 1,172 knee studies, and 7,570 brain boxes plus 643 study-level
labels spanning 30 categories on 1,001 brain studies. The annotations are
single-reader, non-exhaustive indicators, not diagnostic ground truth. The
paper states that the 1,001 brain studies were randomly selected from 5,847
brain studies. This creates an important prior-training leakage risk addressed
in Section 4.

The fastMRI+ conversion flips image arrays vertically for the radiologist
view. The official conversion must be reproduced exactly before mapping a box
back to the reconstruction array. Gate 0 includes a manual overlay audit.

### 3.2 KITTI-360 and R2DM

Use registered access to
[KITTI-360](https://www.cvlibs.net/datasets/kitti-360/). The dataset owners
report more than 100,000 laser scans over 73.7 km and release the data under
CC BY-NC-SA 3.0 for noncommercial use; registration and intended-purpose
declaration are required. The
[official documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
defines the raw Velodyne layout, timestamps, poses, and annotations.

Use the public implementation and weights from the
[official R2DM repository](https://github.com/kazuto1011/r2dm), associated
with the [R2DM paper](https://arxiv.org/abs/2309.09256). The repository
documents a 163 GB KITTI-360 download, an MIT code license, a two-channel
range/reflectance representation, and pretrained weights. Its official dataset
loader fixes the `spherical-1024` representation at 64x1024 and the split:

- R2DM training drives: 3, 4, 5, 6, 7, 9, 10;
- R2DM test drives: 0, 2.

Do not redistribute KITTI-360 scans in the code release. Derived images shown
in the paper must retain the required attribution and share-alike terms.

### 3.3 Immutable provenance manifest

Before any calibration result is viewed, create a restricted manifest with:

- source URL and dataset release/download date;
- accepted-license/DUA version or access-page snapshot identifier;
- raw relative path, byte size, and SHA256;
- canonical acquisition/study or drive/frame ID;
- official split;
- anatomy, sequence/contrast, field strength, scanner metadata when present;
- pathology label vector and box count for MRI;
- timestamp, drive, pose availability, and annotation availability for LiDAR;
- QC status and a reason for every exclusion;
- fold assignment;
- prior-training membership, detector-training membership, calibration
  membership, and locked-test membership.

Commit only a deidentified manifest containing dataset-native public IDs and
hashes if the governing agreement permits it. Otherwise commit a salted
manifest digest and store the full manifest in the restricted project area.
Every result artifact records the manifest digest.

If a stable patient identifier is present under the accepted fastMRI access
terms, all acquisitions from that patient form one group. If it is absent,
the maximum auditable unit is the HDF5 acquisition/study: no slice may cross
folds, and the paper must explicitly state that unique-patient grouping could
not be verified. fastMRI+ describes one series per annotated examination, but
this does not justify inventing a patient identifier.

## 4. Split design and leakage controls

### 4.1 MRI: study/patient-level six-fold design

Construct folds independently for brain and knee:

1. Join fastMRI+ reviewed-file lists, annotations, raw HDF5 metadata, and the
   official fastMRI split by exact filename.
2. Resolve a stable patient group if available; otherwise group by HDF5
   acquisition/study. Never group by slice.
3. Apply QC before randomization: required `kspace` and
   `reconstruction_rss`, native k-space support at least 384x320, native RSS
   reference support at least 320x320, brain acquisition in
   `{AXT1, AXT1PRE, AXT1POST, AXT2, AXFLAIR}` or knee acquisition in
   `{CORPD_FBK, CORPDFS_FBK}`, finite k-space/reference values, unambiguous
   annotation transform, and reconciled prior provenance.
4. With seed 2027, perform group-aware iterative multilabel stratification
   into six near-equal folds. Balance pathology-present versus reviewed-normal,
   every exact official fastMRI+ category indicator, contrast/field strength
   for brain, PD versus PD-FS for knee, and scanner/vendor metadata when
   available. Do not semantically merge labels or force exact balance for
   ultra-rare labels.
5. Freeze fold CSVs and SHA256 before training or calibration metrics are
   viewed.

Use the folds as follows:

| Fold(s) | Approximate share | Permitted use |
|---|---:|---|
| 0, 1, 2 | 50% | train the frozen pathology detector and task-supervised IGS comparator |
| 3 | 16.7% | engineering, hyperparameters, wall-clock matching, threshold selection |
| 4, 5 | 33.3% | locked confirmatory evaluation only |

Before QC, this implies approximately 586/195/391 knee studies and
500/167/334 brain studies for detector training/calibration/locked testing.
The exact counts are whatever the frozen grouped stratification produces.
After freezing, a corrupt or unusable case is marked missing with its reason;
the folds are not rebalanced.

All slices in a locked study are evaluated for global metrics. Box metrics use
all mapped boxes after the predeclared transform. A study, not a slice or a
box, is the independent statistical unit.

### 4.2 Prior-training leakage: fail closed

Domain-specific priors are required. A brain model is never a headline knee
prior, and a knee model is never a headline brain prior.

The
[official Ambient Diffusion MRI implementation](https://github.com/utcsilab/ambient-diffusion-mri)
publishes a supervised fully sampled fastMRI EDM checkpoint and brain training
code. It is useful for smoke testing, but it is not automatically admissible
for confirmatory fastMRI+ brain evaluation. Because the 1,001 annotated brain
studies were sampled from the broader fastMRI brain cohort, checkpoint
provenance must be reconciled against every calibration and locked filename.

The rule is:

- if the checkpoint's exact training filenames and hashes prove disjointness
  from calibration and locked studies, it may be used;
- if exact provenance cannot be recovered, the checkpoint is smoke-test-only;
- for the confirmatory track, train a fresh brain EDM and a fresh knee EDM on
  official training studies after excluding every patient/study assigned to
  folds 3--5;
- save the exact training manifest, source commit, configuration, and
  checkpoint SHA256.

This exclusion applies to prior training, checkpoint selection, learned mask
training, and pathology detector training. Pretraining on a locked case cannot
be excused because the downstream sampler is zero-shot.

### 4.3 LiDAR: drive-level split

Define a primary-eligible frame before selecting timestamps. Its raw Velodyne
file must decode to an \(N\times4\) float array with \(N>0\) and all finite
values; its timestamp, vehicle pose, and official calibration chain must be
present and finite; the pinned R2DM `spherical-1024` projection must produce
exactly 64x1024 cells with at least 1,024 valid returns in [1.45, 80] m.
Resolve raster collisions with the pinned official loader, not a local
heuristic. Semantic or object-label availability is not a primary-eligibility
criterion. Record every failed predicate and never apply reconstruction- or
method-dependent QC.

Three strictly ordered tracks are allowed, and they must not be pooled:

**Pre-S3 engineering smoke.** Use the official R2DM checkpoint only on drive
9, which is calibration data and part of the released checkpoint's training
split. Exercise I/O, projection, budgets, and metric code there; do not report
the numbers as evaluation.

**Confirmatory fresh-prior track.** Train R2DM configuration H from scratch on
drives 3, 4, 5, 6, and 7; use drive 9 for calibration; lock drives 0, 2, and
10 for testing. Select 1,500 approximately uniformly spaced valid frames from
each locked drive, yielding 4,500 frames. This gives three unseen drives and
ensures that the fresh confirmatory checkpoint has not trained on drive 10. The
deterministic selection rule is: sort valid timestamps, map 1,500 evenly
spaced quantiles to the nearest unique indices, and break ties toward the
earlier frame.

**Post-S3 released-checkpoint reproduction.** Only after the confirmatory
configuration and results are immutable, run the official R2DM checkpoint on
up to 2,500 uniformly spaced valid frames from each of drives 0 and 2. Apply
the same timestamp-sort, quantile, and earlier-tie rule and the already frozen
policy and metric configuration without tuning. Report each drive separately
and label this a released-prior reproduction, not an independent confirmatory
result.

No frame from a locked drive can be used to estimate normalization, ray-drop
noise, candidate thresholds, model selection, or wall-clock configuration.
Frames are correlated; statistical resampling must preserve temporal blocks.

### 4.4 Frozen training contracts

Training data are not artificially restricted to the annotated detector
subset when a task-agnostic model can legally use more data. Fresh MRI priors,
LOUPE, and task-agnostic learned active policies may use all otherwise
admissible official training studies, but must exclude every patient/study in
folds 3--5. The task-supervised IGS comparator and pathology detector use only
folds 0--2 because their targets derive from fastMRI+ annotations. The fresh
R2DM prior uses only drives 3--7; drive 9 remains calibration-only.

Before the first S1 training run, write and hash one machine-readable contract
per learned component. Every contract must include the exact source commit,
training manifest digest, preprocessing code/configuration, model
architecture, initialization, optimizer and schedule, effective global batch,
gradient accumulation, precision, update count, EMA rule, checkpoint cadence,
checkpoint-selection rule, training seeds, and final checkpoint SHA256.
Missing fields block the component from S2.

The following recipes are frozen:

| Component | Training recipe | Selection rule |
|---|---|---|
| brain and knee EDM priors | copy the supervised fully sampled R=1 architecture and numerical optimization fields from the pinned [official Ambient MRI training recipe](https://github.com/utcsilab/ambient-diffusion-mri/blob/main/train.sh); use separate anatomy manifests and seed 2027; change only paths, anatomy, and distributed launch mechanics while preserving effective global batch and optimizer trajectory | terminal EMA checkpoint at the predeclared update count copied from the released supervised `training_options.json`; no pathology or reconstruction metric enters selection |
| fresh R2DM prior | run the pinned [official R2DM `train.py`](https://github.com/kazuto1011/r2dm/blob/main/train.py) default configuration H on the drive-3--7 dataset builder with seed 2027 | terminal EMA checkpoint at configuration H's fixed `training.num_steps`; drive 9 is diagnostic only and cannot select the checkpoint |
| task-agnostic learned masks/policies | use the full admissible prior-training pool, identical normalization and legal line actions, and the method's official loss; train seeds 2027, 2028, and 2029 | pair each training seed with the same-numbered inference/replay stream and average those three runs; do not select a seed or form a 3x3 cross-product |
| IGS | folds 0--2 only, with the frozen detector/task loss and legal full-column actions | terminal scheduled checkpoint; fold 3 sets only a published method hyperparameter when its original protocol requires one |
| pathology detector | fully sampled references from folds 0--2, grouped 90/10 into detector-train/internal-validation with seed 2027; use TorchVision's official [`fasterrcnn_resnet50_fpn_v2`](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html), repeat normalized RSS to three channels, retain 320x320 support, and treat every valid reviewed box as one class named `finding` | highest internal-validation class-agnostic mAP50, earliest epoch on ties; fold 3 chooses the one score threshold only |

The configuration hashes, rather than these prose labels, are authoritative.
If exact released supervised R=1 `training_options.json` provenance cannot be
recovered, freeze an explicit replacement configuration before S1 and label
the model a repository-matched reimplementation rather than the released
recipe. Scaling to two A800s may change per-device batch and accumulation but
not effective global batch, learning-rate schedule, number of optimizer
updates, or examples seen.

## 5. Measurement-emulation protocols

### 5.1 Multicoil Cartesian MRI

The primary MRI cell is:

- full multicoil SENSE forward model;
- model support 384x320;
- fixed 320x320 center support for reported RSS metrics;
- nominal initial acceleration 8x with center fraction 0.04;
- 16 additional phase-encode columns;
- four diffusion checkpoints, with four sequential one-column decisions at
  each checkpoint;
- acquisition after denoiser steps 9, 14, 18, and 23 (zero-indexed);
- synthetic complex Gaussian noise at scale 0.01 relative to a clean-ACS
  simulator statistic;
- sampler normalization and sensitivity-map estimation from noisy acquired ACS
  data only.

One action acquires a complete Cartesian phase-encode column: every readout
sample and every coil at that column. The physical budget is the number of
columns, not the number of coils, complex components, or tensor entries. For
each slice:

\[
L_{\mathrm{final}}=L_{\mathrm{initial}}+16,
\]

unless fewer than 16 columns remain, in which case the case fails primary-cell
eligibility rather than silently changing budget. The implementation must
assert the exact initial and final counts.

For the synthetic MRI noise cell, compute
`estimate_acs_rss_scale(mask * clean_kspace, center_fraction=0.04,
quantile=0.99)` once inside the simulator. Give every coil the same complex
noise variance equal to the square of 0.01 times that scale, and draw one
circular complex Gaussian field as
\((\epsilon_\mathrm{re}+i\epsilon_\mathrm{im})/\sqrt{2}\) with the frozen
noise seed. Generate the field once before masking; all policies reveal the
same entries. This clean-ACS quantity sets a known simulator noise level only
and is never an inference-time normalization.

There are 16 feedback opportunities. After each selected column is revealed,
the likelihood and low-dimensional belief are conditioned on that column
before scoring the next column at the same diffusion checkpoint; this
conditioning does not add a denoiser or finite-difference call. The
four-line-packet ablation instead scores and commits four columns from one
pre-packet belief. It is not the headline configuration.

All policies see the same initial mask, noisy measurement field, coil maps,
and stochastic seed for a slice. The complete noisy k-space tensor exists
inside the retrospective simulator but may be indexed only after an action has
been selected. Target RSS, clean k-space, boxes, and target-derived maxima are
evaluation-only. The simulator may use a clean-ACS statistic to define a known
noise level, but deployable normalization is recomputed from acquired noisy
ACS. This distinction must be stated.

Sensitivity maps are estimated from the same noisy ACS for all methods.
Freeze the current low-resolution estimator: a 4% central ACS box, separable
Tukey window with alpha 0.5, inverse FFT, coil-wise division by RSS with
epsilon \(10^{-6}\), and background suppression below 2% of the ACS-RSS
maximum. Normalize inference inputs by the 99th percentile of the noisy
acquired-ACS RSS, never by a target maximum.
A predeclared robustness cell treats coil-map/nuisance uncertainty jointly;
the headline known-map approximation and the joint-nuisance result must not be
mixed.

Secondary measurement cells, run only after the primary configuration is
frozen, are:

| Initial mask | Added columns | Purpose |
|---|---:|---|
| nominal 8x, 0.04 center | 8 | lower adaptive budget |
| nominal 8x, 0.04 center | 16 | **primary** |
| nominal 8x, 0.04 center | 24 | budget curve |
| nominal 4x, 0.08 center | 8 | moderate-acceleration robustness |

Do not choose the paper's headline cell after viewing the locked budget curve.

This remains a retrospective line-selection study. Per-slice adaptive masks
are legal in the emulator and are counted per slice, but the paper must not
claim sequence timing, gradient switching, or scanner deployment without a
prospective implementation.

### 5.2 Programmable LiDAR

Preprocess raw KITTI-360 Velodyne scans with the official R2DM
`spherical-1024` path:

- height 64, width 1024;
- channel 0: normalized range;
- channel 1: reflectance;
- explicit validity/no-return mask;
- valid depth range 1.45--80 m, matching the official loader.

R2DM's 64x1024 tensor is a derived spherical raster, not the native HDL-64E
packet/firing layout. Define one **angular query** as one raster location.
An angular query jointly reveals its encoded range and reflectance values, so
it produces two scalar fields. Do not call a raster query a verified physical
laser firing until a raw packet/ring/azimuth audit establishes a one-to-one
mapping. The programmable-sensing interpretation is therefore a nominal ray
command in the emulator, while the auditable budget unit is the angular
query.

The pinned preprocessing encodes a no-return cell with sentinel -1 inside the
two normalized continuous R2DM channels. Revealing those two values can
therefore reveal a no-return without a third validity bit. The separate source
validity array is retained only as a metric target and is never exposed to the
planner or categorical likelihood.

Every method begins with a fixed scout that observes all 64 elevation rows at
every eighth azimuth column:

\[
64(1024/8)=8{,}192\ \text{angular queries}=12.5\%.
\]

Partition the raster into 4x128 angular tiles. There are
\((64/4)(1024/128)=128\) candidate tiles. Each tile contains 512 angular
locations, of which 64 were already observed by the scout, so one action
reveals exactly:

\[
4(128-16)=448\ \text{new angular queries}
=896\ \text{new scalar fields}.
\]

The budget table is therefore:

| Added tiles | Rounds x tiles | New angular queries | New scalar fields | Total observed queries | Total fraction |
|---:|---:|---:|---:|---:|---:|
| 8 | 4x2 | 3,584 | 7,168 | 11,776 | 17.97% |
| 16 | 4x4 | 7,168 | 14,336 | 15,360 | 23.44% |
| 32 | 4x8 | 14,336 | 28,672 | 22,528 | 34.38% |

Sixteen tiles is the frozen primary budget. The implementation must assert
that every selected action contains 448 previously unobserved angular queries
and that the final query total is 15,360. The same angular mask is applied to
range and reflectance. A supplementary audit maps raw point order, laser ring,
azimuth, projection collisions, and empty bins. Only if that audit validates
the mapping may tables add “nominal rays” in parentheses.

As in MRI, the four diffusion checkpoints contain four sequential tile
decisions each, for 16 feedback opportunities. Reveal one tile, condition the
likelihood and compressed belief, and only then score the next tile at that
checkpoint; no extra denoiser/JVP is permitted between these decisions. A
four-tile packet without within-packet conditioning is a labeled ablation.

The headline study reveals the exact held-out raster encoding after selection.
The source validity mask is retained separately for evaluation; RaoSense's
policy, likelihood, belief update, and reconstruction readout never receive it.
RaoSense's headline likelihood remains limited to the two continuous R2DM
channels and must not be described as a categorical ray-drop model. Its small
per-channel likelihood variance is a numerical regularizer, not injected
sensor noise.

For robustness experiments, create one `observed_scan` by adding the
calibration-fitted heteroscedastic range noise and ray-drop process to hidden
`full_scan`. Use that same `observed_scan` for belief conditioning and data
consistency; never hard-clamp hidden truth. Do not fit a noise model to locked
drives. A simulated dropped return writes sentinel -1 to both continuous
channels. Missing returns must remain explicit: zero is not an automatically
ignorable target. A hybrid validity likelihood is an optional, separately
labeled extension, not part of the current completed implementation.

This is programmable-beam emulation, not hardware validation. It does not
measure MEMS settling, laser power, eye-safety constraints, weather, or
closed-loop latency.

The existing adapter can be exercised with:

```python
from src.samplers.raosense_lidar import (
    build_lidar_tile_actions,
    make_lidar_scout_mask,
    run_raosense_lidar,
)

scout = make_lidar_scout_mask(
    channels=2,
    height=64,
    width=1024,
    elevation_stride=1,
    azimuth_stride=8,
    device=full_scan.device,
)
actions = build_lidar_tile_actions(
    channels=2,
    height=64,
    width=1024,
    tile_height=4,
    tile_width=128,
    observed_mask=scout,
    device=full_scan.device,
)
result = run_raosense_lidar(
    diffusion=r2dm,
    full_scan=full_scan,
    observed_scan=full_scan,
    initial_mask=scout,
    candidate_actions=actions,
    num_steps=32,
    n_branches=8,
    tangent_rank=2,
    active_actions=16,
    active_rounds=4,
    active_after_frac=0.30,
    active_until_frac=0.75,
    planning_horizon=2,
    beam_width=8,
    num_fantasies=16,
    information_samples=128,
    planner_candidate_cap=None,
    measurement_noise_variance=1e-3,
    seed=2027,
)
assert result["nfe"] == 384
assert len(result["acquisitions"]) == 16
assert sorted({a.step for a in result["acquisitions"]}) == [9, 14, 18, 23]
assert all(a.new_scalar_measurements == 896 for a in result["acquisitions"])
assert 8_192 + sum(
    a.new_scalar_measurements for a in result["acquisitions"]
) // 2 == 15_360
```

For the headline cell, set `measurement_noise_variance=1e-3`; the scalar is
broadcast over both channels and all 64x1024 cells. It is a frozen likelihood
regularizer, not an estimate of sensor variance, and it does not inject
measurement noise. Its value lives in the hashed production configuration,
not a separately fitted calibration artifact.

This call is an adapter-level recipe, not yet an end-to-end data command.
Gate 0 requires a production driver that loads the pinned R2DM checkout,
checks drive/frame manifests, converts scalar indices to angular-query counts,
audits any claimed mapping from angular queries to physical firings, runs every
baseline, performs canonical replay, and writes resumable per-frame artifacts.

## 6. Baselines and comparison tiers

### 6.1 Required MRI baselines

At minimum, implement the following on the exact multicoil line action set:

| Family | Method | Role |
|---|---|---|
| Handcrafted | equispaced, variable-density random, adaptive random | acquisition controls and timing control |
| Static learned | [LOUPE](https://arxiv.org/abs/1901.01960), trained separately for brain and knee | strong nonadaptive learned mask |
| Task-aware static | [Iterative Gradient Sampling (IGS)](https://conferences.miccai.org/2022/papers/371-Paper0303.html), trained with only folds 0--2 | reconstruction/pathology-task mask comparator |
| Learned active | [Zhang et al. uncertainty acquisition](https://arxiv.org/abs/1902.03051), [Pineda et al. RL](https://github.com/facebookresearch/active-mri-acquisition), and [Bakker et al. multicoil policy](https://arxiv.org/abs/2203.16392) | three required subject-adaptive learned comparators |
| Diffusion active | [ADS](https://openreview.net/forum?id=OGifiton47) | particle-entropy diffusion baseline |
| Diffusion active | [AdaSense](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10059.pdf) | posterior-covariance adaptive sensing |
| Diffusion BED | [Iollo et al.](https://arxiv.org/abs/2506.16237) | closest diffusion experimental-design comparator |
| In-family | \(K=12,T=32\) point ensemble variance | exact 384-NFE covariance control |
| In-family | \(K=16,T=32\) point ensemble variance | larger point-ensemble quality/memory reference, 512 NFE |
| RaoSense | H=1 and H=2 | paired planning comparison |

If an original method chooses arbitrary Fourier points, project its score onto
legal full phase-encode columns without changing the measurement budget. If
the projection changes the method materially, label it “ported” and release
the adaptation. An external baseline that requires unavailable privileged
inputs is reported in a separate track, not silently weakened.

`operator_diag_active`, `sketch_marginal_active`, and
`sketch_logdet_active` may appear as historical/in-family ablations, but
pre-audit `pv` and `pv_active` are not valid headline covariance baselines.

### 6.2 Required LiDAR baselines

At minimum:

| Family | Method | Role |
|---|---|---|
| Static | scout only, uniform tile lattice, random tile order | budget controls |
| Static learned | calibration-trained fixed tile ranking | tests instance adaptation |
| Active ensemble | R2DM \(K=12\) and \(K=16\) ensemble variance | exact-NFE and larger in-family ensemble controls |
| Published adaptive | [Adaptive LiDAR Sampling using Ensemble Variance](https://arxiv.org/abs/2007.13834) | direct uncertainty baseline |
| Published interactive | [Uncertainty-Aware Interactive LiDAR Sampling](https://openaccess.thecvf.com/content/WACV2023/html/Taguchi_Uncertainty-Aware_Interactive_LiDAR_Sampling_for_Deep_Depth_Completion_WACV_2023_paper.html) | WACV programmable-sampling comparator |
| RaoSense | H=1 and H=2 | paired planning comparison |

[Deep Adaptive LiDAR](https://davidlindell.com/publications/adaptive-lidar)
and
[RL beam-configuration optimization](https://arxiv.org/abs/2201.03860)
are useful task-driven references. Their original RGB guidance, detector
training, or global beam-configuration assumptions are privileged relative to
R2DM-only range-image completion. If implemented, report them in a clearly
labeled privileged-input/task-trained table as well as, if possible, a
range-only port.

### 6.3 Diagnostic oracle

For both domains, compute a hindsight one-step action ranking using the hidden
complete target on a small calibration subset. It measures whether a policy's
scores correlate with realized reconstruction/task improvement. It is labeled
`TARGET-ACCESS DIAGNOSTIC`, never a deployable baseline and never an upper
bound: a one-step target-informed choice need not be globally optimal.

## 7. Fairness: separate four questions

One table cannot fairly answer mask quality, native system quality, exact NFE,
and deployable latency. Report four comparison tiers.

### 7.1 Tier A: prior-conditioned selection-only mask quality

For every policy:

1. run the acquisition policy and retain only its final mask;
2. reconstruct that mask with the same canonical passive, single-trajectory,
   32-step solver;
3. use streams 2027--2029 for all calibration candidates and for the locked
   confirmatory set; use only stream 2027 for ruled-out descriptive methods
   and the \(K=16\) exception, exactly as frozen in Section 2.2;
4. evaluate the canonical reconstruction.

This isolates measurement selection from branch averaging or a method's
native reconstruction, but it remains conditional on the same EDM/R2DM prior
used by RaoSense. Call it “prior-conditioned mask quality,” not
reconstructor-independent mask optimality. In MRI,
`scripts/reconstruct_mc.py` performs this canonical replay unless
`--skip_canonical_replay` is set. The canonical replay cost is 32 additional
NFE and must be recorded separately from policy NFE. The LiDAR driver must
implement the same rule with one fixed 32-step R2DM completion.

As a secondary sensitivity analysis, replay every frozen final mask with an
independent reconstructor: fixed-iteration SENSE-TV for MRI and a deterministic
spherical total-variation/inpainting solver for LiDAR. Tune their
regularization only on calibration data and hold iteration count and stopping
rule fixed. Agreement across prior-conditioned and independent replay supports
a mask claim; disagreement limits the claim to the diffusion prior.

### 7.2 Tier B: native end-to-end systems

Report each method's own reconstruction, MRI-column or LiDAR-angular-query
budget, actual NFE, wall time, peak memory, and learned-training requirement.
Do not interpret a native reconstruction difference as purely a mask
difference.

### 7.3 Tier C: exact-NFE comparison

The exact-NFE headline contains only methods that execute exactly 384 denoiser
calls per sample:

- RaoSense H=1 and H=2: \(8(32)+2(8)(2)(4)=384\);
- point ensemble: \(12(32)=384\);
- a ported ADS/AdaSense/BED configuration only if its measured call counter is
  exactly 384.

Methods below or above 384 are still reported with their actual cost but are
not called equal-NFE. Never pad a fast learned policy with useless denoiser
calls to manufacture equality. Also report denoiser forward batches and
effective samples/second so batching cannot hide calls.

### 7.4 Tier D: wall-clock-matched comparison

Tune wall-clock variants only on MRI fold 3 and LiDAR drive 9. Use the same
two-A800 node, but time one exclusive A800 per process at batch size one;
the second GPU runs an independent shard, not model parallelism. Record:

- exact GPU model and memory, CPU, RAM, storage type;
- CUDA, driver, PyTorch, precision, and deterministic settings;
- ten warm-up samples;
- three repeats of the same deterministic timing subset;
- `torch.cuda.synchronize()` around CUDA events and host timing;
- median, IQR, and 95th-percentile wall time;
- peak allocated and reserved GPU memory;
- planner time and denoiser time separately.

Freeze, for each baseline, the strongest calibration configuration whose
median latency is within +/-5% of RaoSense H=2. If no configuration enters the
window, report the nearest lower- and higher-latency variants. A cheap learned
policy keeps its true lower latency and is not slowed down artificially.

Planner Monte Carlo and candidate screening count toward wall time even though
they do not count as NFE.

## 8. Metrics

### 8.1 MRI global fidelity

Report on the fixed center 320x320 RSS support:

- volume-stacked NMSE;
- volume-stacked PSNR using one target data range per volume;
- mean slice SSIM aggregated within volume first;
- HFEN and gradient-magnitude error;
- all metrics by anatomy, contrast, field strength, and pathology presence.

Because model preprocessing is fixed at 384x320 and evaluation at center
320x320, these are “fixed-support fastMRI-style” metrics, not an official
challenge submission score.

The primary global endpoint for each anatomy is the paired study-level
difference in canonical-replay volume PSNR.

### 8.2 MRI pathology and hallucination

Map fastMRI+ boxes through the documented flip/crop transform and compute:

- box NMSE, PSNR, SSIM, HFEN, and gradient error;
- the same metrics on a size-, slice-, and radial-location-matched normal patch
  from that study;
- lesion-to-matched-normal error ratio;
- locally flagged reconstruction-inconsistency patches with the FDA's
  [specific Fourier ring correlation method](https://cdrh-rst.fda.gov/sfrc-detecting-hallucinations-medical-image-restoration)
  and [official sFRC code](https://github.com/DIDSR/sfrc);
- frozen detector recall, precision, mAP50, and category sensitivity;
- false-positive lesion detections per reviewed-normal study.

The primary pathology endpoint is study-averaged pathology-box normalized
RMSE, oriented so a reduction is positive. Detector and sFRC outcomes are
secondary safety endpoints because the labels are single-reader and
non-exhaustive.

For a valid transformed box \(b\), define

\[
\operatorname{nRMSE}_b =
\sqrt{
\frac{\sum_{p\in b}(\hat{x}_p-x_p)^2}
{\max\{\sum_{p\in b}x_p^2,\,
10^{-12}|b|D_v^2\}}
},
\]

where \(D_v\) is the frozen full-volume data range. Clip boxes to the 320x320
evaluation support and exclude one before the fold freeze if less than 50% of
its original transformed area remains or either clipped side is under four
pixels. Deduplicate identical coordinates/labels. Average boxes equally
within category, then categories equally within study, then compare studies
equally. Use \(D_v\), never a small ROI's own maximum, as the box-PSNR data
range. The confirmatory box population consists only of reviewed studies with
at least one valid image-level box. Diffuse or study-level-only labels enter
the detector analysis, not the box endpoint.

Freeze sFRC on fold 3 separately for brain and knee: 64x64 patches, Hann
windowing enabled, FRC correlation threshold 0.75, native fully sampled
iFT/RSS reference, and an anatomy-specific intensity window fixed from fold 3.
Select the hallucination cutoff using only known calibration ROIs constructed
by controlled removal and insertion of small structures, then lock it before
test. Report flagged patches per megapixel and the fraction overlapping
fastMRI+ boxes. sFRC identifies locally unfaithful patches; it does not by
itself classify a flag as additive or subtractive. Any signed-error
subclassification is exploratory and must be validated separately.

Train the detector only on fully sampled reference reconstructions from folds
0--2, tune thresholds only on fold 3, and freeze it before reconstruction
tests. Never train on method outputs. Its confirmatory readout is the
class-agnostic `finding` endpoint. Report sensitivity stratified by an exact
official category only when the locked test contains at least 30 positive
studies for that category; do not merge rare labels post hoc or report a
macro-average dominated by categories with only a handful of cases.

Before production, two authors independently inspect 25 brain and 25 knee box
overlays sampled with seed 2027. Any coordinate mismatch blocks the pathology
study.

### 8.3 LiDAR completion and task metrics

Let \(v_{ij}\) be the source validity mask. Decode the continuous predicted
range through the pinned R2DM utility before any validity decision. For every
cell with \(v_{ij}=1\), replace a nonfinite decoded prediction by 80 m and
clip the prediction to [1.45, 80] m before computing range MAE/RMSE. Thus a
model-predicted no-return at a true-return cell still receives a range error;
it is never removed from the range denominator. Reflectance MAE uses the same
target-valid cells.

The continuous headline model has no learned validity head. Before S3, use
fresh-prior passive completions on drive 9 to choose one threshold and
direction on the raw normalized range channel that maximizes frame-macro
validity F1. Freeze that scalar in `range_validity_threshold.json` and apply
it to every method and locked drive; method-specific thresholds are
prohibited. Apply the same threshold to the continuous reconstruction at both
queried and unqueried cells. The source validity mask is an evaluation target
only: it is never passed to the acquisition policy, likelihood, belief update,
or readout. This rule preserves the two-scalar-field angular-query budget and
does not turn RaoSense's likelihood into a categorical ray-drop model. Report
return and no-return precision, recall, F1, and IoU over all 64x1024 cells.

Convert target-valid and predicted-valid ranges to vehicle-frame points with
the pinned R2DM ray-angle and `to_xyz` implementation. Apply these exact
geometry rules:

- discard points outside [1.45, 80] m;
- order points by elevation row then azimuth column and, if a cloud exceeds
  16,384 points, retain 16,384 evenly spaced order statistics with ties toward
  the earlier raster cell;
- symmetric Chamfer is the mean of the two directed mean nearest-neighbor
  Euclidean distances, in meters, without squaring;
- point-to-plane is the mean of the two directed absolute normal projections;
  estimate each destination cloud's normals by PCA on its 20 nearest
  neighbors, with deterministic index tie-breaking;
- if exactly one cloud is empty, assign 80 m to both distances; if both are
  empty, assign zero.

The BEV occupancy grid is fixed in the vehicle frame to
\(x,y\in[-80,80)\) m with 0.5 m square cells and
\(z\in[-3,3]\) m. A cell is occupied when it contains at least one valid
point. IoU is intersection over union; define an empty-empty frame as one and
an empty-nonempty frame as zero. Do not use the distributional BEV histogram
metric from R2DM as a substitute for this paired completion IoU.

Also report:

- range MAE in target bands 1.45--20 m, 20--40 m, and 40--80 m;
- errors inside official KITTI-360 3D object primitives and outside them;
- pseudo-label semantic consistency between reconstructed and full-scan
  predictions from one frozen LiDAR segmenter;
- ground-truth semantic mIoU only for frames with an exact, audited projection
  into official KITTI-360 3D labels.

Transform object primitives and semantic labels into the frame's Velodyne
coordinate system using the official timestamp, pose, and calibration chain.
Freeze a `semantic_label_map.json`; official void/unknown IDs and any class
without a one-to-one declared mapping are ignored rather than coerced. Before
S3, two authors inspect object/semantic overlays for 50 drive-9 frames sampled
with seed 2027. Any timestamp, handedness, or coordinate mismatch blocks the
object and semantic analyses.

The primary LiDAR endpoints are valid-return range MAE and BEV occupancy IoU.
No-return F1 is a safety endpoint.

If a segmenter such as the
[official RangeNet++ implementation](https://github.com/PRBonn/lidar-bonnetal)
is used, a prediction-to-prediction comparison is called pseudo-label
semantic consistency, not ground-truth mIoU. Report each drive separately.

## 9. Statistical analysis

### 9.1 MRI

For each metric, first aggregate slices and boxes within a study. Compare
methods by paired study-level differences. For the confirmatory set, average
the three frozen paired streams within study/frame first; single-stream
descriptive and \(K=16\) comparisons remain explicitly single-stream. Use:

- 10,000 paired bootstrap resamples with seed 2027;
- two-sided percentile 95% confidence intervals for every comparison;
- one-sided paired randomization tests for preregistered superiority
  hypotheses, with alpha 0.05;
- study-level effect size and the full difference distribution;
- Holm correction across the two primary endpoint p-values within each
  dataset;
- Holm correction across brain, knee, and LiDAR for the H=2-versus-H=1
  planning claim;
- Benjamini--Hochberg FDR at 0.05 for clearly labeled secondary endpoints.

An endpoint passes only when its ordinary two-sided 95% interval excludes zero
in the favorable direction and its one-sided Holm-adjusted p-value is below
0.05. Brain-and-knee generality is an intersection-union claim: both
anatomy-specific conjunctions must pass. This protocol does not use the
undefined shorthand “Holm-adjusted confidence interval.”

Select a nonoracle comparator separately for each primary endpoint using only
calibration data: highest mean PSNR, lowest mean pathology nRMSE, lowest mean
LiDAR range MAE, and highest mean BEV IoU. If calibration means differ by less
than 0.05 dB, 1% relative error, or 0.5 absolute IoU percentage points as
applicable, break the tie by lower NFE and then lower wall time. Freeze six
dataset-endpoint comparator identities: PSNR and pathology nRMSE for each MRI
anatomy, plus range MAE and BEV IoU for LiDAR. Each primary test compares
against its own frozen endpoint comparator; the union of selected methods
appears in the main table. No comparator can be reselected from locked
results.

Whenever a gate says “noninferior” or “no harm,” use these margins rather
than absence of a superiority p-value. With differences oriented as
RaoSense-minus-comparator, the one-sided 95% paired lower confidence bound
must exceed -0.05 dB for MRI PSNR, -0.5 absolute percentage points for BEV
IoU, and -0.5 absolute percentage points for no-return F1. For error metrics,
the one-sided 95% upper confidence bound on the paired geometric-mean ratio
RaoSense/comparator must be below 1.01 for pathology nRMSE and LiDAR range
MAE. Use study resampling for MRI and the frozen moving-block resampling for
LiDAR. Equality at a margin does not pass.

No per-slice p-value is permitted. A box count is not a sample size.

### 9.2 LiDAR

Frames in a drive are temporally correlated. Use a hierarchical moving-block
bootstrap. Freeze its duration from calibration data as follows:

1. sort the selected drive-9 frames by original timestamp;
2. form two frame-level series: median valid range and valid-return fraction;
3. compute each series' absolute sample autocorrelation at integer frame lags;
4. take the first lag for which both autocorrelations stay below 0.1 for five
   consecutive lags, convert it to the corresponding median elapsed time, and
   set the block duration to the larger of that duration and five seconds;
5. if no such run occurs within the first quarter of the calibration sequence,
   use that quarter-sequence duration and flag the persistence in the report.

For each locked drive, resample contiguous frames in original timestamp order
until the calibrated duration is reached, using circular wraparound, and
concatenate blocks until the original frame count is restored. Compute paired
method differences within each resampled drive, weight drives equally rather
than by frame count, and repeat 10,000 times with seed 2027. Use the same
blocks for every method and metric. The one-sided LiDAR p-values use a paired
blockwise randomization test at this same frozen duration. Report sensitivity
analyses at half and twice the duration; these do not replace the primary
result.

With the official checkpoint there are only two test drives. Report
drive-specific effects and explicitly avoid population-of-drives inference.
The fresh-prior track adds a third locked drive but is still a small number of
environments.

A drive split is not proof of novel geography. Before S3, use official poses
to compute, for every locked frame, the minimum planar distance to any
prior-training or calibration frame and report the fractions within 5, 10,
and 25 m, plus the corresponding heading differences. The primary result
remains conditional on drives 0, 2, and 10. A predeclared sensitivity analysis
excludes, without replacement, locked frames within 10 m of any prior-training
frame. If the effect vanishes there or too few frames remain, do not claim
geographic generalization.

### 9.3 Missingness and numerical failures

Every failure is retained in an audit table. Primary analysis is
intention-to-evaluate using the fixed failure scores below. A secondary
complete-case analysis may be shown. If a method fails on more than 1% of
locked samples, it fails the reliability gate regardless of its mean score.

Freeze failure scores before S3 at the independent-unit level: 0 dB for MRI
PSNR, nRMSE 10 for the MRI pathology endpoint, 80 m for LiDAR range MAE, and
0 for BEV IoU. Any failed slice makes that MRI method/study a failure unless a
bitwise-identical retry from the saved pre-slice state succeeds; any failed
frame makes that method/frame a failure. Do not define a penalty from the
worst locked result.

### 9.4 Calibration-only power and precision check

Before opening S3, use only calibration paired differences and the already
frozen locked-manifest counts to simulate 10,000 studies of the planned tests.
Report power under the substantive effects in Gates 3 and 4 and the expected
95% interval half-width. The pathology simulation uses only locked studies
known from their manifest to contain at least one valid box; it does not use
their reconstructed outcomes.

Proceed with the confirmatory pathology endpoint only if each anatomy has at
least 80% simulated power to resolve a 5% relative nRMSE reduction at the
Holm-adjusted family alpha. Proceed with the LiDAR task claim only if the
conditional-drive design has at least 80% power for either a 10% range-MAE
reduction or two BEV-IoU points. If a check fails, revise the claim and
protocol before opening S3; do not reinterpret an underpowered negative
result post hoc.

## 10. Experiment matrix

### 10.1 Stages

| Stage | Data | Purpose | Test access |
|---|---|---|---|
| S0 | analytic/toy | correctness and falsification | complete synthetic access; final clean rerun PENDING |
| S1 | 20 MRI studies/anatomy from folds 0--2; 200 drive-9 LiDAR frames | end-to-end smoke, exact budget/NFE assertions, storage profiling | no locked data |
| S2 | MRI fold 3, LiDAR drive 9 | tune only frozen hyperparameters, select comparator, wall-clock match, fit noise | no locked data |
| S3 | full locked primary cells | brain, knee, and LiDAR confirmatory endpoints | once |
| S4 | frozen robustness cells and predeclared ablations | mechanism and stress tests | after S3 configuration freeze |

S3 does not start until every Gate 0 artifact exists. Partial S3 results are
not used to alter a method.

### 10.2 Primary table

For each of brain, knee, and LiDAR, the primary table contains:

- static/equispaced;
- random adaptive;
- best learned nonadaptive or active baseline;
- calibration-selected diffusion-active baselines (union across endpoints);
- exact-NFE \(K=12\) ensemble;
- larger in-family \(K=16\) ensemble;
- RaoSense H=1;
- RaoSense H=2.

Columns contain MRI columns or LiDAR angular queries, NFE, wall time, peak
memory, canonical primary metrics, native primary metrics, and reliability
failures.

### 10.3 Predeclared ablations

Run broad ablations on calibration data. Only the bold primary setting and the
mechanism contrasts below are repeated on locked data.

| Factor | Values |
|---|---|
| branches \(K\) | 1, 4, **8**, 16 |
| tangent rank \(q\) | 1, **2**, 4 |
| covariance decomposition | between only, within only, **total law of covariance** |
| planning horizon | H=1, **H=2** |
| fantasies | 4, **16**, 64 |
| beam width | 4, **8**, 16 |
| candidate cap | 24, 48, **all** |
| belief correction | off, **on** |
| acquisition rounds | 1, 2, **4** |
| operator treatment | fixed/known, joint nuisance uncertainty |
| MRI action grouping | one line sequentially; four-line packet without within-packet conditioning |
| LiDAR tile geometry | 2x256, **4x128**, 8x64 at equal 512-location tile area |
| LiDAR prior | official checkpoint smoke; **fresh disjoint R2DM** |
| MRI prior | public brain smoke; **fresh disjoint anatomy-specific EDM** |
| reconstruction readout | canonical single trajectory; weighted branch mean |
| measurement corruption | noiseless retrospective; calibrated noise/ray drop |

The within-only and between-only policies use identical branches, probes,
rounds, and seeds; only the covariance term changes. H=1 and H=2 restart from
identical branch/probe streams. This pairing is required to identify the
algorithmic contribution.

## 11. Preregistered go/no-go gates

These thresholds are decision rules, not promises of positive results.

### Gate 0: integrity and executability

All of the following are required:

- data access complies with the current agreements/licenses;
- the fastMRI citation language, image-publication permission, derived-artifact
  release decision, and post-project destruction plan are documented;
- immutable patient/study and drive manifests exist;
- prior, learned-policy, detector, calibration, and locked-test memberships
  are disjoint as specified;
- every raw input and checkpoint has a verified hash;
- every learned component has a complete, frozen training-contract hash;
- tracked worktrees and external source checkouts are clean and pinned;
- the final analytic suite and targeted tests pass;
- exact MRI-column, LiDAR-angular-query, and NFE assertions pass on all S1
  samples;
- H=1/H=2 common-random-number and zero-future equivalence tests pass;
- LiDAR `observed_scan` separation regression passes and no hidden target is
  used by conditioning or data consistency;
- the confirmatory H=2 root set is uncapped;
- fastMRI+ box overlays pass two-author inspection;
- LiDAR object/semantic overlays and pose-overlap audits pass;
- MRI pathology output retention and the production LiDAR driver are
  implemented and resumable;
- no method has target access before selecting an action.

Failure blocks the main run.

### Gate 1: synthetic mechanism

At the final evidence commit:

- Rao--Blackwell covariance MSE is lower than point-particle covariance MSE;
- the information estimate lies inside its analytic sandwich;
- measured sufficient rank does not exceed the theoretical cap;
- H=2 changes the root action in the routing construction;
- all targeted tests pass.

The fresh analytic artifact and full 43-test CPU suite both pass at
code-under-test commit `03746c938bd09e1bad620353d851ac32818d4ece`, whose
history includes paired zero-future equivalence and LiDAR observation
separation. Gate 1 is numerically satisfied at that code commit, but final
evidence archival remains PENDING until this protocol is committed and the
test log plus artifact hashes are retained at one clean evidence commit.

### Gate 2: calibration action quality

Freeze `gate2_states.csv` before computing any policy score. For each anatomy,
order fold-3 primary-eligible studies by SHA256 of
`"2027|" + study_id`, and take the first 64. Sort that study's slice indices
that pass the already frozen finite-value/support QC and choose the lower
median, index \(\lfloor(n-1)/2\rfloor\). Retain the **round-entry** belief
immediately before the first of four sequential actions at each checkpoint:
256 states per anatomy. For LiDAR, take 256 evenly spaced quantiles of
primary-eligible drive-9 timestamps and retain the same four round-entry
states: 1,024 states. A state is geometry-eligible only when at least four
legal root actions remain and every root has at least one legal follow-up.
This predicate is evaluated before any score or target loss. A nonfinite
policy/diagnostic value is a Gate-2 failure, not an exclusion.

Within each frozen state, correlate scores over every legal root with the
target-access diagnostic's realized canonical loss reduction; assign
Spearman correlation zero to a constant score vector. Aggregate state
correlations with equal study/frame weight. Proceed only if:

- total RaoSense score improves Spearman correlation by at least 0.05 over
  between-only particle covariance in at least two of brain, knee, and LiDAR;
- its paired bootstrap interval is positive in at least one domain;
- H=2 changes the H=1 root action on at least 10% of eligible states in at
  least one domain;
- the equal-unit mean reduction in realized two-action target-access regret
  for H=2 relative to H=1 has a strictly positive paired 95% lower confidence
  bound in at least one domain.

If not, stop and diagnose the belief before spending the locked-test budget.

### Gate 3: MRI locked efficacy

For both brain and knee, RaoSense H=2 must beat the frozen
measurement-matched nonoracle comparator for each endpoint on:

1. canonical volume PSNR; and
2. pathology-box normalized RMSE.

For both endpoints in each anatomy, the ordinary two-sided 95% interval must
exclude zero favorably and the one-sided Holm-adjusted p-value must be below
0.05. In addition, the result must be substantive, not merely powered by case
count:

- at least +0.20 dB PSNR in one anatomy; and
- at least 5% relative pathology-box error reduction or +2 percentage points
  frozen-detector lesion recall in one anatomy;
- failure rate at most 1%.

If only one anatomy succeeds, the paper cannot claim anatomy-general MRI
benefit.

### Gate 4: LiDAR locked efficacy

At the 16-tile primary budget, RaoSense H=2 must beat the frozen
measurement-matched nonoracle comparator for each of valid-return range MAE
and BEV occupancy IoU: each ordinary two-sided 95% interval must exclude zero
favorably and each one-sided Holm-adjusted p-value must be below 0.05. It must
also achieve at least one substantive threshold:

- 10% relative reduction in range MAE; or
- +2 absolute IoU points in BEV occupancy.

The one-sided paired lower confidence bound for the no-return-F1 difference
must exceed -0.5 absolute percentage points under the frozen noninferiority
rule. Failure rate must be at most 1%.

### Gate 5: compute and representation value

RaoSense H=2 must:

- beat the \(K=12,T=32\) point-ensemble mask under exact 384 NFE in at least
  two of the three datasets and meet every applicable Section-9.1
  noninferiority margin in the third; and
- versus the larger in-family \(K=16,T=32\) ensemble, use exactly 75% of its
  NFE.

It must also be Pareto-useful: either improve primary quality with no more than
25% wall-time overhead, or match quality while reducing wall time by at least
20%. For this gate, “improve primary quality” means that the paired ordinary
95% interval favors RaoSense on at least one primary endpoint and meets the
noninferiority margins below on every other primary endpoint in the datasets
supporting the claim. “Match quality” means all corresponding one-sided 95%
paired confidence bounds meet these frozen noninferiority margins:

- MRI PSNR: no worse than -0.05 dB;
- MRI pathology nRMSE: no more than 1% relatively higher;
- LiDAR range MAE: no more than 1% relatively higher;
- LiDAR BEV IoU: no worse than -0.5 absolute percentage points.

Quality and timing in the \(K=16\) comparison use paired inference stream 2027
for both methods; the three-stream RaoSense estimate is not compared against a
single-stream ensemble estimate. If H=2's planner is much slower and quality
does not compensate, the efficiency claim is dropped.

Total peak memory includes the shared denoiser and can obscure representation
cost because branches are partly sequential. Report both total peak memory and
the increment over a loaded-model/passive-sampler baseline. Claim a memory
advantage only if both measures are at least 20% lower than the \(K=16\)
reference; memory is otherwise descriptive, not a failed efficacy gate.

### Gate 6: nonmyopic value

Against paired H=1, H=2 must have a favorable two-sided 95% interval and a
Holm-adjusted one-sided p-value below 0.05 on a primary endpoint in at least
one domain, meet every applicable Section-9.1 noninferiority margin in the
other domains, and achieve at least one of:

- +0.10 dB MRI PSNR;
- 2% relative pathology-box error reduction;
- 2% relative LiDAR range-MAE reduction;
- +1 BEV IoU point.

If this gate fails, remove “nonmyopic” and horizon-two superiority from the
title, abstract, and contributions. A synthetic routing example alone is not
enough.

### Gate 7: generality and submission decision

- **Full operator-general submission:** Gates 0--6 pass, with positive primary
  evidence in brain, knee, and LiDAR.
- **Narrow MRI submission:** both anatomies and Gates 0, 1, 3, 5, and 6 pass,
  but LiDAR does not; remove cross-operator claims.
- **Belief-only revision:** H=2 fails but the total Rao--Blackwell belief
  strongly beats ensembles in multiple domains; remove the planning claim and
  rerun the power/novelty assessment.
- **No-go:** the method fails exact-NFE comparisons, succeeds only on one
  dataset, relies on leakage/target access, or produces clinically/task-relevant
  degradation despite a small PSNR gain.

No budget, metric, baseline, or anatomy can be selected post hoc to rescue a
failed gate.

## 12. A800 compute and storage plan

All numbers in this section are planning estimates, not measured results. S1
profiling replaces them before the expensive run.

### 12.1 Storage

| Item | Source-backed or planned size |
|---|---:|
| fastMRI knee multicoil train + validation | 1.123 TB from the dataset paper |
| KITTI-360 for R2DM | 163 GB from the official R2DM repository |
| fastMRI brain | measure exact selected release after authorized download; do not infer from case count |
| R2DM/EDM checkpoints and processed caches | plan 0.3--0.8 TB |
| metrics, masks, logs, selected float16 reconstructions | plan 0.2--0.5 TB |

Provision at least 4 TB if only the annotated brain subset is local and 6 TB
for a safer knee + brain + KITTI + cache workspace. A full fresh brain prior
may require additional object storage; size it from the verified download
manifest before training.

Do not save every branch state at every step. Persist:

- per-sample metrics and diagnostics;
- final masks and branch weights;
- selected qualitative cases fixed before viewing method identity;
- float16 or lossless visualization outputs as appropriate;
- enough failure state to resume deterministically.

### 12.2 GPU planning envelope

Assuming two 80 GB A800 GPUs:

| Work package | Planning estimate |
|---|---:|
| fresh brain EDM | 3--6 A800-GPU days |
| fresh knee EDM | 3--6 A800-GPU days |
| fresh disjoint R2DM | 3--5 A800-GPU days |
| smoke + calibration + wall-clock sweep | 2--5 A800-GPU days |
| locked primary baselines and RaoSense | 20--40 A800-GPU days |
| predeclared ablations and robustness | 10--20 A800-GPU days |
| total envelope | 41--82 A800-GPU days |

The locked-run row must include three complete inference streams for every
endpoint comparator, \(K=12\), H=1, and H=2, plus the separate seed-2027
\(K=16\) reference; canonical replay is additional. The broad range above is
a reservation, not evidence that these multipliers have been profiled. S1
must produce a per-method spreadsheet that expands every stream, anatomy,
slice/frame, NFE, replay, and retry allowance before S3 is authorized.
The calibration row must likewise include three paired runs for every required
comparator candidate, using the seed-pairing rule in Section 2.2.

On two GPUs this is approximately three to six wall-clock weeks at high
utilization. It can be reduced only by measured profiling, fewer redundant
methods, or early gate failure--not by dropping locked cases after seeing
results.

After S1, estimate each job from measured batch-one synchronized latency:

\[
\text{GPU days} =
\frac{\sum_d N_d \,\bar{s}_d\, \sum_m t_{d,m}}
{86{,}400},
\]

where \(N_d\) is studies/frames in dataset \(d\), \(\bar{s}_d\) is mean
evaluated slices per MRI study (one for LiDAR), and \(t_{d,m}\) is median
seconds per slice/frame for method \(m\). Add 20% for restarts, I/O, and
artifact generation. Freeze this revised budget before S3.

Use the two GPUs for independent volume/frame shards with deterministic shard
manifests. Merge only after verifying unique sample keys and identical config
hashes.

## 13. Execution recipes and required implementation work

### 13.1 Final synthetic rerun

```bash
python scripts/validate_raosense.py \
  --seed 2027 \
  --covariance_trials 300 \
  --particles 12 \
  --action_dimension 4096 \
  --information_samples 10000 \
  --output outputs/raosense/final_analytic_validation.json

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
```

Record `git rev-parse HEAD`, `git status --porcelain`, Python/package versions,
the complete test log and SHA256, and the analytic-output SHA256.

### 13.2 MRI manifest

`scripts/make_fastmri_cohort_manifest.py` already verifies explicit filenames,
anatomy, dimensions, and SHA256. It intentionally does not auto-select files.
However, its current default applies the 384x320 minimum to both k-space and
RSS. Before production, split this into a 384x320 k-space requirement and a
320x320 RSS-reference requirement. This is not permission to synthesize or pad
the target.

Generate one immutable manifest per anatomy and evidence stage from the frozen
fold list. At scale, add a reviewed `--filenames_file` input rather than
passing hundreds of basenames through shell globbing.

The current script writes `split="validation"` and
`scripts/reconstruct_mc.py` requires that literal value for headline runs.
That schema is not sufficient for this protocol: the fastMRI+ paper identifies
the 1,172 knee cases as validation studies but does not make the same statement
for all 1,001 brain cases. Before production, replace the single field with
both `official_source_split` and `protocol_split`, where the latter is one of
`detector_train`, `calibration`, or `locked_test`, and allow the runner to
validate the derived split. Never relabel an official brain training file as
“validation” to satisfy the current assertion. Keep the six-fold assignment
in the frozen companion manifest.

### 13.3 MRI primary run

The sampler flags for the in-repository primary and exact-NFE controls exist,
but the current loader rejects an RSS reference smaller than the 384x320 model
support. Standard fastMRI references can be 320x320. Gate 0 therefore requires
one production fix before this command is valid: reconstruct on native
384x320-or-larger k-space support, center-crop the reconstruction to 320x320,
and compare it directly with the native 320x320 RSS reference. Do not enable
`--allow_spatial_padding` and do not synthesize RSS from the coil images.

After that fix, the primary command is:

```bash
python scripts/reconstruct_mc.py \
  --data_path "$FASTMRI_BRAIN_H5" \
  --dataset_revision "$FASTMRI_REVISION" \
  --volume_manifest "$BRAIN_LOCKED_MANIFEST" \
  --anatomy brain \
  --num_volumes "$N_BRAIN_LOCKED" \
  --minimum_production_volumes 300 \
  --num_slices 0 \
  --target_resolution 384 320 \
  --eval_size 320 \
  --acceleration 8 \
  --center_fraction 0.04 \
  --mask_seed 2027 \
  --noise_scale 0.01 \
  --noise_seed 2027 \
  --schedule edm \
  --num_steps 32 \
  --sigma_min 0.002 \
  --sigma_max 80 \
  --checkpoint_dir "$BRAIN_CHECKPOINT" \
  --checkpoint_manifest "$BRAIN_CHECKPOINT_MANIFEST" \
  --adps_root "$PINNED_ADPS_ROOT" \
  --expected_adps_commit "$ADPS_COMMIT" \
  --active_lines 16 \
  --active_rounds 4 \
  --active_after_frac 0.30 \
  --active_until_frac 0.75 \
  --ens_particles 12 \
  --rao_branches 8 \
  --rao_tangent_rank 2 \
  --rao_beam_width 8 \
  --rao_fantasies 16 \
  --rao_information_samples 128 \
  --rao_candidate_cap 0 \
  --sketch_fd_step 0.05 \
  --methods static_match random_adaptive equi_adaptive \
            ens_active ens_random_active \
            raosense_greedy_active raosense_h2_active \
  --seed 2027 \
  --device cuda \
  --output_dir "$BRAIN_OUTPUT"
```

This command's ensemble control is \(K=12\). For the nonlearned \(K=12\),
H=1, H=2, and any nonlearned endpoint-comparator job, produce streams 2027,
2028, and 2029 by changing only `--seed` and the output directory; mask and
measurement-noise seeds remain 2027 so every stream sees the same
observations. For a learned comparator, also switch the learned-policy
checkpoint and checkpoint manifest to the matching training seed. No other
argument changes. This implements the three paired, not 3x3, rule in
Section 2.2.

Run the larger \(K=16\) representation reference as a separate seed-2027 job
made from the complete frozen command above with exactly these substitutions:
`--ens_particles 16`, `--methods ens_active ens_random_active`, and a new
`--output_dir`. Its expected online cost is \(16(32)=512\) NFE. The artifact
merger must reject any attempt to combine \(K=12\) and \(K=16\) shards under
one configuration hash. Gate 5 compares this job only with RaoSense stream
2027.

Run the identical command for knee after changing only the anatomy-specific
data manifest, fresh knee checkpoint/manifest, volume count, and output path.
Do not use `--skip_file_hashes`, `--allow_spatial_padding`,
`--allow_missing_rss_target`, `--legacy_full_scan_noise`, or
`--skip_canonical_replay` for evidence.

Before S3, the MRI path still requires:

1. separate k-space/model-support and RSS/evaluation-support validation as
   described above;
2. official-source and derived-protocol split fields instead of the current
   hard-coded `split="validation"` assertion;
3. reconstruction retention or streaming metric hooks for pathology boxes,
   matched normal patches, sFRC, and the frozen detector;
4. a group-aware six-fold manifest builder;
5. external LOUPE/IGS/learned-active/ADS/AdaSense/BED ports;
6. joint coil-map/nuisance ablation;
7. resumable per-volume shards and a strict artifact merger;
8. exact CUDA event timing and planner/denoiser time separation.

### 13.4 LiDAR production driver

The driver must expose one config file containing:

```yaml
dataset:
  name: kitti360
  representation: spherical-1024
  manifest_sha256: PENDING
  calibration_drives: [9]
  locked_test_drives: [0, 2, 10]
measurement:
  observation_mode: exact_encoded_replay
  validity_side_channel: false
  scout_elevation_stride: 1
  scout_azimuth_stride: 8
  tile_height: 4
  tile_width: 128
  active_tiles: 16
  active_rounds: 4
  continuous_likelihood_variance: 0.001
sampler:
  steps: 32
  branches: 8
  tangent_rank: 2
  horizon: 2
  beam_width: 8
  fantasies: 16
  information_samples: 128
  candidate_cap: null
  active_after_frac: 0.30
  active_until_frac: 0.75
  seed: 2027
```

It must fail unless the angular-query budget is 15,360, NFE is 384, the R2DM
source/checkpoint hashes match, and no locked drive appears in prior training
or calibration. It must write canonical and native reconstructions/metrics,
query masks, selected tiles, return-validity metrics, wall time, memory, and
failure state.

As for MRI, run the endpoint comparator, \(K=12\), H=1, and H=2 with paired
streams 2027--2029 while keeping the initial scout and retrospective
observation fixed. Run \(K=16,T=32\) in a distinct seed-2027 configuration
with an asserted 512 NFE and its own output/configuration hash. The production
config must also name the frozen validity-threshold, geometry-metric,
semantic-map, and block-duration files described in Sections 8 and 9.

## 14. Artifact and paper checklist

Every row in a paper table must be traceable to:

- repository Git commit and clean/dirty status;
- external source commits;
- raw manifest digest and exact split;
- checkpoint and checkpoint-training-manifest digests;
- full command/config hash;
- method seed streams;
- MRI-column or LiDAR-angular-query count and scalar-field count;
- NFE, denoiser batches, wall time, and peak memory;
- canonical versus native reconstruction label;
- independent-unit count and failure count;
- bootstrap seed, block size, multiplicity family, and confidence interval;
- an explicit status: `PILOT`, `CALIBRATION`, or `LOCKED`.

Qualitative figures are selected by a rule frozen before method identities are
revealed: median case, 10th-percentile case, 90th-percentile case, largest
pathology-box disagreement, and largest method failure for each domain.
Cherry-picked “best-looking” examples are supplementary only and labeled.

Release code, configs, manifests where licensing permits, metric tables, and
derived masks only after a per-artifact licensing review. Keep fastMRI-derived
case variables, hashes, masks, reconstructions, and metric files restricted
unless NYU clears their release. Keep KITTI-derived reconstructions, masks,
and figures separate from the software license and apply the dataset's
attribution, noncommercial, and share-alike terms. Do not release restricted
raw measurements. State prominently that fastMRI+ annotations are
non-exhaustive single-reader labels and that the LiDAR study is retrospective
programmable-beam emulation.

## 15. Primary-source index

Data, priors, and evaluation:

- [fastMRI data portal](https://fastmri.org/)
- [official fastMRI code and format](https://github.com/facebookresearch/fastMRI)
- [fastMRI knee dataset paper](https://pubs.rsna.org/doi/10.1148/ryai.2020190007)
- [fastMRI+ paper](https://www.nature.com/articles/s41597-022-01255-z)
- [fastMRI+ annotations](https://github.com/microsoft/fastmri-plus)
- [Ambient Diffusion MRI official code/checkpoints](https://github.com/utcsilab/ambient-diffusion-mri)
- [FDA sFRC description](https://cdrh-rst.fda.gov/sfrc-detecting-hallucinations-medical-image-restoration)
- [FDA sFRC code](https://github.com/DIDSR/sfrc)
- [KITTI-360 portal and license](https://www.cvlibs.net/datasets/kitti-360/)
- [KITTI-360 documentation](https://www.cvlibs.net/datasets/kitti-360/documentation.php)
- [R2DM official repository](https://github.com/kazuto1011/r2dm)
- [R2DM paper](https://arxiv.org/abs/2309.09256)
- [RangeNet++ official implementation](https://github.com/PRBonn/lidar-bonnetal)

Comparator sources:

- [LOUPE](https://arxiv.org/abs/1901.01960)
- [Iterative Gradient Sampling paper](https://conferences.miccai.org/2022/papers/371-Paper0303.html)
- [Iterative Gradient Sampling code](https://github.com/cviaai/IGS/)
- [uncertainty-driven active MRI](https://arxiv.org/abs/1902.03051)
- [active MRI with reinforcement learning](https://github.com/facebookresearch/active-mri-acquisition)
- [adaptive multicoil MRI policies](https://arxiv.org/abs/2203.16392)
- [Active Diffusion Subsampling](https://openreview.net/forum?id=OGifiton47)
- [AdaSense](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10059.pdf)
- [diffusion-guided Bayesian experimental design for MRI](https://arxiv.org/abs/2506.16237)
- [Deep Adaptive LiDAR](https://davidlindell.com/publications/adaptive-lidar)
- [adaptive LiDAR ensemble variance](https://arxiv.org/abs/2007.13834)
- [WACV 2023 uncertainty-aware interactive LiDAR](https://openaccess.thecvf.com/content/WACV2023/html/Taguchi_Uncertainty-Aware_Interactive_LiDAR_Sampling_for_Deep_Depth_Completion_WACV_2023_paper.html)
- [RL LiDAR beam configuration](https://arxiv.org/abs/2201.03860)
