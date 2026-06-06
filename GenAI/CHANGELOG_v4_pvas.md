# v4 — SENSE forward, PV-gated active sampling (PVAS)

**Date:** May 4, 2026
**Commits:** `6d8201b`, `fa68f41`, `373781b`, `371c59d`, `ba11402`,
`2aa0735`, `e576ffb`, `d0d93f7`, `507aff6`, `1883b86`, plus several Colab
fixes (`a005471`, `741ef79`, `9242a94`, `6e77387`, `783e6b0`, `0a0ba75`).

## What changed

- **SENSE forward operator** (`feat(multicoil)` in `6d8201b`) with
  per-coil-per-frequency FA-KGD weighting — the multicoil generalization of
  the FA-KGD update.
- **PV-gated FA-KGD-MC + active line acquisition** (`fa68f41`): use the
  posterior-variance map from the sampler as a budgeted acquisition
  function for the next k-space line.
- Active baselines added (`371c59d`): `random_adaptive`, `equi_adaptive`,
  `energy_active`, `oracle_active` so the active-sampling comparison is
  honest.
- PV gate diagnostics surfaced in `reconstruct_mc` summaries
  (`K_div_vs_pigdm`, `P_ang_frac`) for ablation transparency.
- `paper/pvas/` draft with figure pipeline (`scripts/generate_pvas_figures.py`)
  and navy colorway.
- Several Colab/runtime fixes: lazy `s3fs` install, ADPS auto-clone, device
  fixes in `_add_kspace_noise`, slice zero-padding before center-crop.

## GenAI usage

**What was AI-assisted**
- SENSE forward boilerplate (per-coil FFT, coil-combination, adjoint).
  I provided the math; AI generated the einsum-heavy PyTorch and I tested
  the adjoint property numerically before trusting it.
- Active-baseline scoring functions (`random_adaptive`, `equi_adaptive`
  templates).
- Figure-generation script (`generate_pvas_figures.py`) — matplotlib
  styling, axis ticks, the navy palette.
- LaTeX table macros for the PVAS paper draft.
- Diagnosing several Colab-specific runtime errors (s3fs missing, ADPS
  not cloned, CUDA-vs-CPU mismatch in `_add_kspace_noise`, slice-size
  edge case).

**What was human-led**
- The PV-gating idea itself — using the sampler's intermediate posterior
  variance as an acquisition signal rather than a fixed mask.
- Decision to include `oracle_active` as a topline upper-bound baseline.
- All claims, ablation choices, and figure compositions in the PVAS paper
  draft. AI helped with prose polish; the experimental narrative is mine.

## Representative prompts

> "Implement a SENSE forward operator in PyTorch. Inputs: image
> `x [B, 2, H, W]` (real/imag), sensitivity maps `S [B, C, 2, H, W]`,
> sampling mask `M [B, H, W]`. Output: undersampled multicoil k-space
> `y [B, C, 2, H, W]`. Then implement its adjoint. Include a unit test
> that checks `<A x, y> ≈ <x, A^* y>` to 1e-5."

> "I want an active-sampling baseline that scores unsampled k-space lines
> by the *radially averaged* posterior variance from my sampler at a given
> timestep, then picks the top-k. Sketch the function signature consistent
> with my existing `acquisition_fn(state) -> indices` interface, and
> include flags `--pv_n_probes`, `--no_pv_radial_smooth`,
> `--no_pv_normalize` for ablation."

> "My `_add_kspace_noise` errors with `Expected all tensors to be on the
> same device` when run on CUDA. The offending tensor is `sig_c`. Suggest
> the minimal fix (no broad refactor)."
> *(Implemented and committed as `783e6b0`.)*
