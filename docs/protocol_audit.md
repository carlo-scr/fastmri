# Research and Protocol Audit

**Audit date:** 2026-07-25
**Scope:** every local and remote Git branch, tracked paper/prototype code,
ignored experiment outputs, and the active-sensing/covariance literature.

## Verdict

The repository contains promising exploratory behavior, but no existing PVAS
number is currently safe to put in a conference paper. The output files are
real; the problem is interpretation. A corrected rerun is mandatory because
the MRI estimator, diffusion-noise convention, and several controls changed.

Status for all pre-audit PVAS outputs: **LEGACY-INVALIDATED**.

This does not mean the effect is absent. It means the current files may guide
the next experiments but cannot support causal or statistical claims.

## Branch audit

The historical branches `paper-writeup`, `dev/multicoil-sense`,
`presentation/milestone-2`, and `carlopaperwrite` are all ancestors of
`main`. The remote cloud-data feature branch had already been deleted and was
pruned. The substantive uncommitted PVAS-v2 prototype was found on `main`,
preserved on `research/sketchsense-wacv27`, and checkpointed at `dbc5714`
before corrections began.

## Mathematical blockers

### 1. The legacy MRI score was not posterior variance

The implementation used

\[
  \sigma_t^2\left|FJ_tF^\ast z\right|^2.
\]

Its expectation is a squared-Jacobian row energy,
\(\sigma_t^2\operatorname{diag}(BB^\ast)\) with
\(B=FJ_tF^\ast\), rather than
\(\operatorname{diag}(F\sigma_t^2J_tF^\ast)\). Even for a diagonal positive
Jacobian it squares the eigenvalues.

The corrected cross-Hutchinson estimator is

\[
  \operatorname{Re}\!\left[
    \overline z\odot \sigma_t^2FJ_tF^\ast z
  \right].
\]

Analytic regression tests must include a non-unit linear spectrum so the two
quantities cannot accidentally agree.

### 2. The candidate operator omitted multi-coil SENSE

The legacy estimator operated in one Fourier basis and copied its map across
coils. The actual measurement bank is

\[
  (Bx)_c=\mathcal F(S_cx).
\]

Acquisition must therefore probe \(B^\ast\), push the denoiser JVP through
\(B\), and aggregate the resulting coil/readout coordinates into candidate
phase-encode columns.

### 3. Complex EDM noise was mis-scaled

The EDM checkpoint was trained with standard deviation \(\sigma_t\) in each
real network channel. The complex sampler generated

\[
  (\epsilon_{\rm re}+i\epsilon_{\rm im})/\sqrt 2,
\]

giving each channel standard deviation \(\sigma_t/\sqrt 2\) while conditioning
the network on \(\sigma_t\). The diffusion sampler now omits this division.

Measurement noise is a separate convention: division by \(\sqrt2\) remains
correct when \(\sigma_c^2=\mathbb E|n_c|^2\).

## Experimental blockers

### 4. Particle controls did not share one measurement state

Natural-image random ensemble particles previously acquired different masks
and fresh noise. Their mean therefore used a much larger union of
measurements. All particles now share the same fixed noisy measurement field
and atom ranking.

### 5. Static and adaptive random were not timing-only controls

The methods previously used different final random masks. The corrected
protocol creates one random ranking of the initial mask’s complement and
either reveals it up front or across scheduled rounds. Overlapping atoms read
the same fixed noisy field.

### 6. Correction timing differed between single and ensemble methods

New ensemble measurements affected the current step, while single-trajectory
measurements first affected the next step. All policies now apply an immediate
correction after acquisition.

### 7. Selection and posterior averaging were confounded

An ensemble-selected mask reconstructed by an ensemble mean is not directly
comparable with a single-particle selected mask reconstructed once. Production
runs must report:

- a canonical matched solver for every final mask, isolating selection; and
- optional system-level ensemble reconstructions, reported separately.

### 8. Reported cost was not measured

The draft called unequal-compute runs “matched NFE.” The EDM wrapper now
counts every denoiser call and stores the observed delta per method. New
tables must say either “equal NFE” or “compute-accounted frontier” according to
the actual runs.

### 9. Slice-level inference was pseudoreplicated

Masks, latents, and probes were reused across adjacent slices, and slice-level
Wilcoxon tests treated correlated slices from one volume as independent. New
seeds are derived from `(global seed, volume, slice)` and shared across
methods. Statistical inference must aggregate at the volume level or use
cluster-aware resampling.

### 10. The MRI residual oracle was not an upper bound

It accessed the complete future noisy k-space realization and greedily ranked
current residual energy. This is a useful retrospective **greedy oracle
reference**, not a bound on final reconstruction utility.

### 11. The noise estimate used future data

Per-coil noise was estimated from corners of the complete scan before
undersampling. A prospective active-acquisition experiment must use a known
simulation noise level or an acquisition-valid calibration measurement. The
legacy full-scan estimate may only appear as an explicitly labeled
retrospective ablation.

### 12. Measurement budgets could silently drift

Rounded duplicate acquisition steps could drop budget, and zeroing observed
scores could reacquire tied atoms. Schedules now require unique steps,
selection indexes the available set directly, final masks are counted, and
budget assertions fail loudly.

### 13. The natural-image estimator introduced avoidable bias

Clipping each channel/probe before atom aggregation increases positive bias,
and zero-padded smoothing depresses border uncertainty. Probes and channels
are now averaged before one positivity projection, with reflect padding.

### 14. The 256-pixel dataset provenance was unsafe

The CelebA-HQ loader used a training stream, silently fell back to another
dataset, and cached only by requested count. Any headline natural-image run
must record the exact dataset/revision, split, IDs, hashes, and preprocessing,
and must fail closed when the requested source is unavailable.

### 15. Joint design initially ignored the acquired set

Masking acquired atoms out of the candidate list is not equivalent to
conditioning the information objective on them. If an acquired atom and a
candidate atom expose the same uncertainty direction, an unconditioned
log-determinant selector can buy that direction twice. The corrected
implementation initializes its low-rank determinant state with the Gram
contributions of every acquired atom and scores \(F(M\cup S)-F(M)\). The
marginal low-rank control uses the same acquired-set conditioning but does not
update within the new batch.

### 16. Same-probe diagonal completion was not a residual estimator

The Nyström products and cross-Hutchinson diagonal use the same probes.
Subtracting the resulting low-rank diagonal from that cross-Hutchinson
estimate and clipping is therefore a PSD completion heuristic, not an
independent unbiased estimate of the residual diagonal. The headline
SketchSense path now uses a pure PSD low-rank factor. Cross-Hutchinson remains
a separate diagonal-only control; any residual estimator must use independent
probes and report their extra denoiser calls.

## Claim corrections

- The two-trajectory “law of total variance” score is a heuristic combination
  of local Tweedie uncertainty and between-trajectory disagreement unless its
  conditional-sampling assumptions are established. It is not an exact
  posterior decomposition in the current sampler.
- “Oracle” policies are references, not guaranteed upper bounds.
- A few fixed slices do not establish cross-domain generality.
- A Tweedie Jacobian evaluated along a data-guided trajectory is a local
  covariance surrogate, not automatically the exact covariance conditioned on
  all acquired measurements.
- Frequency scheduling and frequency-aware posterior guidance are crowded
  areas; the new paper must be operator-general and correlation-aware.

## Release criteria for a validated result

A result becomes **VALIDATED** only when its artifact records:

1. Git commit and complete command/configuration;
2. dataset source, revision, split, sample/volume IDs, and hashes;
3. deterministic per-sample mask, sampler, probe, and noise seeds;
4. actual final measurement count and actual denoiser NFE;
5. canonical reconstruction outputs for the selected mask;
6. per-image or per-volume metrics, not only aggregate means;
7. paired intervals or volume-level resampling where appropriate;
8. a marker that the run used the corrected complex-noise and operator-space
   implementations.
