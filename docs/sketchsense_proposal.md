# SketchSense: WACV 2027 Research Proposal

**Working title:** *Operator-Space Diffusion Covariance Sketches for
Correlation-Aware Active Sensing*

**Target:** WACV 2027, Round 2, Algorithms track

**Status (2026-07-25):** method implementation and falsification stage. The
checked-in legacy PVAS outputs are exploratory evidence only and must not be
used as paper results; see `docs/protocol_audit.md`.

## 1. The paper in one sentence

SketchSense uses a few denoiser Jacobian-vector products from one live,
data-guided reverse-diffusion trajectory to build a positive-semidefinite
local surrogate for the covariance of *feasible future measurements*, then
uses its off-diagonal correlations to select a non-redundant measurement
batch conditional on the measurements already acquired.

The paper is not another frequency schedule. The implementation supports a
general linear candidate-measurement bank and is instantiated with grouped
Cartesian SENSE lines and natural-image blocks. Nonlinear sensing would
require an explicit operator JVP/VJP linearization and is outside the current
claim.

## 2. Exact contribution boundary

The narrow claim we may test is:

> We introduce an ensemble-free active-diffusion mechanism whose conditional
> multi-atom utility is computed from an operator-pushed, single-state JVP
> sketch rather than posterior ensembles or a trained acquisition policy.

This is intentionally a *combination* claim. Tweedie covariance, randomized
Nyström approximation, log-determinant experimental design, and Kalman
conditioning are established ideas.

Claims that are forbidden:

- “first diffusion-based active sensing,” “first training-free active
  sensing,” or “first mid-trajectory active sensing”;
- “first posterior-variance acquisition” or “first D-optimal sensor
  placement”;
- “first single-trajectory/JVP/Tweedie covariance estimator”;
- “exact posterior covariance” for an imperfect learned denoiser;
- “distribution-free” or “multimodal posterior coverage”;
- “matched NFE” unless the compared runs have the same measured denoiser-call
  count.

ADS already performs mid-trajectory active diffusion with particles.
AdaSense already performs zero-shot projected-variance acquisition across
multiple operators. Free Hunch already estimates low-rank denoiser covariance
without extra NFE. Christoffel-DPS already studies offline and online sensor
placement for diffusion posterior sampling. Those papers are baselines and
constraints on the claim, not omissions to work around.

## 3. Method

Let the current diffusion denoiser output be

\[
  \mu_t = \mu_\theta(x_t,t),\qquad
  \Sigma_t \approx \sigma_t^2 J_t,\qquad
  J_t = \frac{\partial\mu_\theta(x_t,t)}{\partial x_t},
\]

in the real coordinate representation used by the network. Let \(B\) map an
image to the complete bank of candidate measurements, before selecting a
subset. The local predictive covariance is

\[
  C_t = B\Sigma_t B^\ast.
\]

This is a Tweedie/denoiser local-moment surrogate evaluated at a state shaped
by earlier data-consistency updates. It is not, without further calibration,
the exact \(\operatorname{Cov}(x_0\mid x_t,y_M)\) conditioned on the acquired
measurement set \(M\). Comparison with conditional particle covariance is a
mandatory calibration experiment.

For complex operators such as MRI, every statement is interpreted in stacked
real and imaginary coordinates. A complex-coordinate diagonal is the sum of
its real and imaginary marginal variances and is directly comparable with
\(\mathbb E|n_i|^2\).

### 3.1 Corrected diagonal estimator

For a probe \(z\) in measurement space,

\[
  u=B^\ast z,\qquad
  J_tu \approx
  \frac{\mu_\theta(x_t+\epsilon u,t)
       -\mu_\theta(x_t-\epsilon u,t)}{2\epsilon}.
\]

With exact operator products and independent probes, the cross-Hutchinson
average

\[
  \widehat{\operatorname{diag}}(C_t)
  = \operatorname{Re}\!\left[
      \overline z\odot \sigma_t^2 B J_t B^\ast z
    \right]
\]

is unbiased for \(\operatorname{diag}(B\sigma_t^2J_tB^\ast)\). Finite
differences, clipping, smoothing, and PSD projection introduce approximation
bias. The legacy MRI score

\[
  \sigma_t^2\left|B J_t B^\ast z\right|^2
\]

instead estimates a squared-Jacobian row energy. It is not a covariance
diagonal and cannot support the previous Kalman interpretation.

For multi-coil MRI, \(B\) is the complete SENSE bank
\((Bx)_c=\mathcal F(S_cx)\), not a single Fourier transform copied across
coils.

### 3.2 PSD operator-space sketch

Draw \(q\) probes \(\Omega\) in candidate-measurement space and compute

\[
  X=\Sigma_tB^\ast\Omega,\qquad
  Y=BX=C_t\Omega .
\]

Central finite differences provide the columns of \(X\). Form the Nyström
approximation

\[
  \widehat C_t =
  Y(\Omega^\ast Y)_+^\dagger Y^\ast = LL^\ast,
\]

with an eigenvalue floor in the small \(q\times q\) core. The headline
implementation uses the pure PSD low-rank factor
\(\widehat C_t=LL^\ast\). The cross-Hutchinson diagonal is a separate
ablation. Reusing the same probes to set
\(d=[\widehat{\operatorname{diag}}(C_t)-\operatorname{diag}(LL^\ast)]_+\)
would be a PSD completion heuristic, not an unbiased residual estimate, so it
is disabled. A residual may be added only with independent probes and
explicit extra-NFE accounting.

The implementation must report diagnostics rather than silently forcing a bad
learned Jacobian into this model:

- core eigenvalues and retained rank;
- cross-Hutchinson negative mass before clipping;
- projected asymmetry where it can be estimated;
- explained diagonal fraction and cross-Hutchinson disagreement;
- sensitivity of rankings to the PSD projection.

### 3.3 Correlation-aware grouped acquisition

Candidate atoms may contain several scalar measurements: all coil/readout
coordinates in one MRI phase-encode column, or all channels/pixels in one
natural-image block. Define

\[
  F_t(A)=\frac12\log\det\!\left(
    I+\Gamma_A^{-1/2}\widehat C_{t,AA}\Gamma_A^{-1/2}
  \right).
\]

Given the acquired atom set \(M\), a prospective batch \(S\) is scored by

\[
  \mathcal I_t(S\mid M)=F_t(M\cup S)-F_t(M).
\]

Here \(\Gamma_A\) is the measurement-noise covariance. Greedy selection uses
low-rank determinant updates initialized by the Gram contribution of \(M\).
The marginal sketch baseline also conditions on \(M\), but does not update
after each new atom in the prospective batch. This is the load-bearing
distinction from marginal ranking: correlations should prevent a batch from
buying several copies of the same uncertainty direction.

Already observed linear measurements that are not candidate atoms contribute
an explicit initial Gram matrix. In natural-image super-resolution, this is
how the always-observed box-averaged image conditions both sketch policies;
it is not approximated as a set of full-resolution blocks.

The same sketch can later support a low-rank Kalman correction, but this
becomes a headline contribution only if an acquisition-only ablation shows a
reliable gain. The minimum viable paper is the measurement-space sketch plus
joint design.

### 3.4 Guarantees under the ideal surrogate

The paper should state and prove the limited guarantees the composition
actually supports:

1. if \(C_t\succeq0\), the protected Nyström construction produces
   \(\widehat C_t=LL^\ast\succeq0\) with rank at most \(q\);
2. with independent Gaussian measurement noise, \(F_t(A)\) is normalized,
   monotone, and submodular over grouped atoms, so conditional greedy
   selection has the standard \(1-1/e\) cardinality guarantee for the
   *approximate local model*;
3. if the whitened covariance error on a set of \(m\) scalar measurements is
   at most \(\delta\) in spectral norm, the corresponding log-determinant
   error is at most \(\frac m2\log(1+\delta)\).

These statements do not validate the local Tweedie surrogate itself. Large
core asymmetry, negative mass, or disagreement with conditional particles is
an explicit failure regime.

## 4. Closest work and required distinctions

| Work | What it already establishes | What SketchSense must demonstrate |
|---|---|---|
| [AdaSense, ECCV 2024](https://arxiv.org/abs/2407.08256) | Zero-shot projected posterior-variance acquisition across faces, MRI, and CT using repeated posterior samples | Comparable selection at much lower compute, plus explicit correlated batch selection |
| [ADS, TMLR 2025](https://openreview.net/forum?id=OGifiton47) | Training-free, mid-trajectory active diffusion using parallel particles | One live trajectory plus a few JVPs; matched-budget cost/quality frontier; correlations beyond marginal entropy |
| [Free Hunch, ICLR 2025](https://openreview.net/forum?id=4JK2XMGUc8) | Low-rank denoiser covariance estimates without extra NFE | Whether direct candidate-operator sketches improve acquisition; Free-Hunch-driven selection is a mandatory ablation |
| [Christoffel-DPS, 2026](https://arxiv.org/abs/2605.06861) | Offline/online diffusion sensor placement and non-asymptotic theory; critique of Gaussian OED | A narrower local-Gaussian, operator-grouped, compute-efficient result that survives its baseline |
| [Posterior-DDIM, 2026](https://arxiv.org/abs/2607.19333) | SNR-aware switching in singular directions for a fixed operator | Active design is complementary; use as a passive reconstruction control where practical |
| [Covariance-aware sampling, 2026](https://arxiv.org/abs/2605.13910) | One-JVP structured covariance for diffusion sampling | No JVP or frequency-structure novelty claim; acquisition is the distinction |
| [FGPS, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Thaker_Frequency-Guided_Posterior_Sampling_for_Diffusion-Based_Image_Restoration_ICCV_2025_paper.html) | Frequency-guided posterior sampling for restoration | Avoid positioning the paper as a new frequency-domain schedule |

## 5. Falsifiable paper hypotheses

H1. On analytic non-idempotent Gaussian covariances, cross-Hutchinson
converges to the correct projected diagonal while the legacy squared-JVP score
converges to squared row energy.

H2. At fixed extra NFE, a rank-\(q\) PSD local sketch recovers useful candidate
correlation structure and operational-oracle acquisition rankings better than
a diagonal-only estimate.

H3. When at least eight atoms are selected in a round, joint log-det selection
beats marginal top-variance selection because it avoids redundant atoms.

H4. The same rank and acquisition schedule work in both multi-coil MRI and a
natural-image operator without acquisition-policy training or test-time
finetuning. Each domain still uses its own pretrained diffusion prior.

H5. SketchSense lies on a better cost/quality frontier than particle
uncertainty: fewer NFEs and lower memory for similar or better acquisition
quality. This is not an equal-NFE claim unless an explicit equal-NFE arm is
run.

## 6. Go/no-go gates

These thresholds are predeclared engineering decisions, not paper results.

1. **Mathematics:** analytic tests must pass to numerical tolerance.
2. **Calibration:** rank 4 or 8 must improve Spearman correlation with an
   operational oracle by at least 0.05, or have a clearly positive paired
   confidence interval, over the corrected marginal estimator.
3. **Acquisition:** for batch size at least eight, log-det should gain at least
   0.2 dB in one headline domain and have a positive paired interval in both.
   A weaker second-domain quality effect is acceptable only with a decisive
   compute/memory frontier there.
   The MRI primary cell is fixed in advance at \(R=4,L=8\); the remaining
   grid is robustness evidence, not an opportunity to select the best effect.
4. **Compute:** the headline configuration should use at most about \(1.5\times\)
   passive NFE and less than one quarter of an eight-particle baseline’s NFE.
5. **PSD validity:** if projection changes rankings substantially or the
   learned Jacobian has large negative/asymmetric components, do not present
   the sketch as a posterior covariance.
6. **Baseline:** if Free Hunch or Christoffel-DPS dominates at comparable cost,
   drop estimator novelty and retain only a correlation-aware acquisition
   claim if that ablation still wins.
7. **Scale:** the result must survive volume-level MRI statistics and at least
   250 held-out 256-pixel natural images. CIFAR-only evidence is insufficient.
   These primary intervals condition on the frozen per-sample seed schedule;
   submission additionally requires a multi-seed stability ablation.

If gates 2–3 fail by the first week of August, reposition the project or do
not submit it as a WACV Algorithms paper. A corrected diagonal-only PVAS
fallback is not automatically novel enough given AdaSense, ADS, and Free
Hunch. Do not combine unrelated ideas merely to preserve the SketchSense
title.

## 7. Experiment matrix

### 7.1 Analytic validation

- Real and complex Gaussian priors with exact, non-diagonal covariance.
- Dimensions 64–1024 and ranks 1–32.
- Report diagonal relative error, Frobenius/subspace error, log-det error,
  negative mass, and selected-set regret.
- Explicitly compare cross-Hutchinson, legacy squared-JVP, and pure low-rank
  Nyström SketchSense. Any diagonal completion must be labeled heuristic or
  use independent probes.

### 7.2 Natural images

- **Development:** CIFAR-10 block reveal and guided super-resolution using the
  public DDPM checkpoint.
- **Headline:** the canonical 50,000-file ILSVRC2012 ImageNet validation
  layout, with 250 deterministically selected file IDs and hashes disjoint
  from a reserved 16-image calibration prefix. The loader fails closed on
  subsets, renamed files, training data, or silent fallback.
- Operators: grouped pixel blocks and, if time permits, Fourier masks. One
  well-powered natural-image operator is preferable to several tiny demos.
- Metrics: PSNR, SSIM, LPIPS, NFE, wall time, peak memory, and per-image paired
  intervals.

### 7.3 Multi-coil MRI

- fastMRI brain Cartesian acquisition, \(R\in\{4,8\}\),
  \(L\in\{8,16\}\).
- The predeclared MRI primary cell is \(R=4,L=8\); the other three settings are
  robustness and compute-frontier checks.
- Target at least 20 volume-disjoint validation volumes (12 is a pilot
  minimum); infer at the volume level, not by treating adjacent slices as
  independent.
- Headline metrics use every reconstructed slice in each selected volume so
  RSS NMSE and PSNR are computed on the stacked volume. Because the current
  prior reconstructs fixed 384-by-320 inputs, we use the fastMRI formulas on
  a declared center-320 square crop rather than claiming native-width
  challenge evaluation. Center-slice subsets are pilot-only.
- Per-slice masks, diffusion noise, and probes vary deterministically while
  remaining shared across methods.
- The headline rank-4 configuration uses one acquisition round, giving
  \(T+2q=28\) versus \(T=20\) passive NFEs. Multiple rounds are an explicitly
  higher-cost ablation, not a mislabeled \(1.5\times\) run.
- The known simulation noise level or an acquisition-valid calibration scan
  supplies \(\Gamma\). Do not estimate it from unacquired full-scan corners.
- Per-slice scaling is the 99th percentile RSS of the noisy acquired ACS,
  while the clean-ACS statistic is used only to define controlled synthetic
  noise power. Full-target maxima are evaluation-only.
- Optional prior-shift stress test: brain prior on knee. It is a limitation
  test, not a substitute for the in-domain headline.

### 7.4 Mandatory policies

- static matched-budget random and variable-density masks;
- random adaptive acquisition with the same final ranking as the timing
  control;
- corrected cross-Hutchinson marginal variance;
- PSD sketch with marginal scoring;
- PSD sketch with joint log-det scoring;
- ensemble covariance with \(K\in\{4,8,16\}\);
- faithful ADS and AdaSense baselines separately, plus a clearly labeled
  shared-pipeline Monte Carlo covariance control;
- diffusion-guided MRI Bayesian experimental design (Iollo et al.) where the
  public implementation supports the protocol;
- Free-Hunch-derived selection;
- Christoffel-DPS where its operator support permits;
- retrospective residual/error references, never called upper bounds; on a
  small subset, an exhaustive canonical final-utility oracle.

Every active mask must also be reconstructed with the same canonical solver.
Report the system-level ensemble mean separately so selection value is not
confounded with posterior averaging.

## 8. Planned paper assets

1. **Method figure:** \(B^\ast\Omega\rightarrow\) denoiser JVP
   \(\rightarrow X,Y\rightarrow\) PSD sketch \(\rightarrow\) acquisition.
2. **Estimator table:** probes/rank, extra NFE, diagonal/subspace error,
   Spearman/top-k overlap, PSD diagnostics.
3. **Main table:** measurements, actual NFE, wall time, memory, PSNR, SSIM,
   LPIPS for MRI and 256-pixel natural images.
4. **Ablation table:** diagonal versus pure low-rank; marginal versus
   log-det; acquisition-only versus sketch DC; rank \(q\); batch size; refresh
   schedule.
5. **Frontier figure:** quality versus NFE and versus measurement budget.
6. **Qualitative figure:** masks, covariance correlations, and matched-solver
   reconstructions in both domains.

## 9. Execution calendar

- **July 26–28:** corrected estimator, analytic tests, protocol freeze.
- **July 29–August 2:** rank-2/4 sketch and log-det pilots on CIFAR and about
  20 MRI slices; primary go/no-go.
- **August 3–9:** method freeze and serious baseline integration.
- **August 10–16:** A100 production runs and volume-level analysis.
- **August 17–20:** complete draft, tables, failures, and limitations.
- **August 21:** Round 2 paper registration deadline.
- **August 21–24:** internal review; only predeclared ablations.
- **August 25–27:** anonymization, reproducibility, PDF, and supplement QA.
- **August 28:** paper submission.
- **August 30:** supplementary-material submission.

Official schedule and formatting requirements:
[WACV 2027 dates](https://wacv.thecvf.com/Conferences/2027/Dates) and
[WACV 2027 author guide](https://wacv.thecvf.com/Conferences/2027/AuthorGuides).

## 10. Evidence policy

- **VALIDATED:** produced after the corrected math/protocol commit, with
  config, code commit, dataset revision, IDs/hashes, actual NFE, and per-sample
  outputs recorded.
- **LEGACY-INVALIDATED:** real exploratory output produced before the
  correction; useful for hypothesis generation only.
- **CITED:** copied from a source and clearly attributed.
- **PENDING:** experiment is specified but no number is shown.

No hypothesized number goes into a table, abstract, or result paragraph.
