# v1 — Repository scaffold & oracle baseline

**Dates:** Apr 17 – Apr 18, 2026
**Commits:** `6be7570` → `fe0e94b`

## What changed

- Initial repository layout (`src/samplers/`, `src/models/`, `src/data/`,
  `scripts/`, `notebooks/`, `external/adps/`).
- ΠGDM baseline sampler and a first cut of FA-KGD with the FPDC schedule.
- Oracle-denoiser experiments in `notebooks/04_closed_loop_em_simulation.ipynb`
  to validate the FA-KGD update rule in isolation from a real score network.
- Brain validation data downloaded; first oracle PSNR numbers
  (ΠGDM ≈ 60.3 dB, FA-KGD+FPDC ≈ 63.0 dB at R=4).
- First IEEE-format paper skeleton (`paper/fakgd/`).

## GenAI usage

**What was AI-assisted**
- Repo layout boilerplate (empty `__init__.py` files, argparse skeleton in
  `scripts/reconstruct.py`, basic README structure).
- PyTorch FFT helper functions (`fft2c`, `ifft2c`, RSS combine) — generated as
  starting drafts, then verified against the standard fastMRI conventions.
- LaTeX preamble and IEEE template wiring.
- Stack-trace debugging for the first `ADPS` checkpoint unpickling errors
  (`dnnlib.util` import path).

**What was human-led**
- The FA-KGD/FPDC mathematical formulation (Kalman gain
  $K_i(t) = \sigma_t^2/(\sigma_t^2 + \hat\sigma_i^2)$ and the
  $(\text{step}/(T-1))^\beta$ low-pass schedule).
- Experimental design for the closed-loop EM simulation: noise-vs-radius
  profile, η, β, α_ema settings.
- Interpretation of the +2.7 dB oracle gap as a sampler-quality result rather
  than an end-to-end claim.

## Representative prompts

> "I have a brain MRI complex image of shape `[B, 2, H, W]` (real/imag stacked).
> Write a numerically stable centered 2D FFT pair (`fft2c`, `ifft2c`) using
> `torch.fft.fft2` with orthonormal normalization."

> "Here is the traceback from `pickle.load` on the ADPS supervised EDM
> checkpoint: `ModuleNotFoundError: No module named 'torch_utils'`. What is
> the standard fix without modifying the pickle?"
> *(Used the suggestion — sys.path prepend of `external/adps/` — after
> verifying the ADPS repo layout matched.)*

> "Critique this LaTeX figure caption for an IEEE 2-column paper; flag any
> claims that look stronger than the numerics support."
