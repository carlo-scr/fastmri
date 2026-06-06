# v2 — EDM integration & first realistic results

**Dates:** Apr 18 – Apr 28, 2026
**Commits:** `6810d92` → `12ba2de`, `8b8cf6e`

## What changed

- Wired the ADPS supervised EDM score network (65M-param SongUNet, 384×320,
  2-channel complex) into `src/models/edm_loader.py` with `EDMDenoiser` /
  `OracleDenoiser` wrappers sharing a common interface.
- Added EDM σ-schedule (geometric ρ=7, σ_min=0.002, σ_max=80) in
  `src/samplers/schedules.py`.
- `scripts/reconstruct.py` extended to drive the EDM sampler end-to-end
  with `--method {pigdm,fakgd}` and `--T` sweep support.
- First real-network numbers on knee data (brain-trained domain mismatch):
  R=4: ΠGDM 31.23 ± 0.73 vs FA-KGD 31.39 ± 0.74 (Δ +0.17 dB).
- Diagnosed and **fixed the inverted FPDC schedule bug**:
  `(1 − step/T)^β` → `(step/(T−1))^β` so ACS-only DC at high noise,
  full k-space at low noise (instead of the inverse, which destroyed PSNR).
- Started the volumetric ACS pooling idea (`8b8cf6e`).

## GenAI usage

**What was AI-assisted**
- EDM σ-schedule code (geometric interpolation between σ_min/σ_max with the ρ
  parameter), translated from the EDM paper's formulas.
- `dnnlib`/`torch_utils` import-shim diagnosis for the ADPS pickle.
- Argparse plumbing for the growing flag surface in `reconstruct.py`.
- Markdown/Plotly cells in `notebooks/05_edm_evaluation.ipynb` for the first
  PSNR/SSIM tables.

**What was human-led**
- The FPDC inversion debug. I noticed by inspection that the early-step
  reconstructions were collapsing; I traced the wrong direction of the
  schedule by hand and then asked Copilot to confirm the corrected
  formulation matched my intent.
- Choice to evaluate the brain-trained model on knee data as an honest
  out-of-domain stress test rather than silently swapping checkpoints.
- All reported PSNR/SSIM values (computed by my code, on my machine, against
  fastMRI ground truth using `src/utils/metrics.py`).

## Representative prompts

> "Given EDM's noise schedule
> $\sigma_i = (\sigma_{\max}^{1/\rho} + i/(N-1)(\sigma_{\min}^{1/\rho} -
> \sigma_{\max}^{1/\rho}))^\rho$, write a PyTorch function that returns the
> full descending σ sequence for N steps. Include the trailing $\sigma_N = 0$
> step."

> "My ΠGDM PSNR drops by ~15 dB when I enable my FPDC mask. Current schedule
> is `mask_radius = r_max * (1 - step/T)**beta`. Walk me through whether the
> direction of the schedule is consistent with the intuition that early
> diffusion steps are at *high* noise and should only enforce data
> consistency on the most reliable (low-frequency) measurements."
> *(This conversation surfaced the inversion; I implemented the fix.)*

> "Refactor `reconstruct.py` so the choice of denoiser
> (`oracle`|`edm`) and sampler (`pigdm`|`fakgd`) are separate factory
> functions; preserve the existing CLI flags."
