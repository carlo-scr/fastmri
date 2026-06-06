# v3 — Multicoil ACS, M-step clamp, hyperparameter sweeps

**Dates:** Apr 28 – May 3, 2026
**Commits:** `490876d`, `e44d4c5`, `c04c35a`, `5ed4f29`, `8992217`,
`812bfc5` → `46310b5`, `3dd9280`

## What changed

- Multicoil-ACS noise estimator (`src/samplers/acs.py`) with two modes:
  - `per_slice_acs` (degenerate in single-coil RSS — flagged explicitly in
    the paper)
  - `pooled_acs` (volumetric pooling across N_z slices; ~17× MSE reduction
    on 16-slice brain volumes, validated in
    `notebooks/06_volumetric_acs_pooling.ipynb`)
- Robust ACS noise estimator: min-of-corners + K-slice pooling for outlier
  resistance (`c04c35a`).
- Per-R `center_fraction` in the sweep harness so the ACS region doesn't
  saturate at high acceleration (`e44d4c5`).
- **M-step stabilization** — added `m_step_mode ∈ {full, clamp, off}` to
  `src/samplers/fakgd.py`. Diagnosed that with α_ema=0.95 and EDM σ_max=80,
  α_t ≈ 0.9988 throughout, so one-sided innovations ratchet σ_i² upward,
  driving K→0 and killing data consistency. `clamp` prevents σ_i² from
  exceeding its initial estimate; `off` freezes σ_i².
- Colab-driven hyperparameter sweep (`notebooks/colab_run_sweep.ipynb`,
  `scripts/sweep_hyperparams.py`) over (β_fpdc, α_ema) and over R.
- `scripts/run_acs_pooling_sweep.sh` for the Step-4 oracle-vs-pooled
  ablation.

## GenAI usage

**What was AI-assisted**
- Drafts of `acs.py` — sliding-window noise variance estimators in the
  corner regions of k-space, then a min-of-corners reducer. I specified the
  algorithm; AI wrote the vectorized PyTorch.
- Colab sweep cell scaffolding (idempotent clone+pull, parameter grid loop,
  result aggregation to JSON).
- Pandas/Matplotlib plotting code for the sweep summary figures.
- Suggesting `m_step_mode="clamp"` after I described the runaway-σ_i² symptom.

**What was human-led**
- The diagnosis of the M-step instability (computing α_t at EDM σ_max=80 by
  hand, recognizing the one-sided innovation pathology, deciding that with
  a well-calibrated multicoil_acs init the M-step has nothing useful to do).
- All choices about which sweep points to run, what to compare against, and
  which results to keep vs. discard.
- The interpretation that single-coil per-slice ACS is *degenerate*
  (1 replicate per pixel) — and the editorial decision to make that an
  explicit contribution rather than hide it.

## Representative prompts

> "Write a vectorized PyTorch function that estimates a per-frequency noise
> variance map from a 2D k-space tensor by sampling four corner patches of
> size `patch=12` and taking the min variance (to be robust to anatomical
> energy leaking into the corners). Input `[B, H, W]` complex, return
> `[B, H, W]` real positive."

> "I have an EM-style update `sigma2 += alpha * (innovation^2 - sigma2)` that
> is supposed to refine per-frequency noise variances during diffusion. With
> α_t close to 1 and innovations clamped to be non-negative, sigma2 only
> ever grows. What stabilization options are standard, and which preserves
> the M-step semantics when the initial estimate is already good?"

> "Generate a Colab cell that runs a 3×4 grid of `(beta_fpdc, alpha_ema)`
> through `scripts/reconstruct.py --method fakgd`, collects the per-volume
> PSNR/SSIM into a dict, writes to `outputs/sweep_summary.json`, and
> renders a heatmap with matplotlib. Make it resumable (skip configs
> already in the JSON)."
