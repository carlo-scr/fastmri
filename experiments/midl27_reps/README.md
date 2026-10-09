# MIDL 2027 pilot: repetitions as supervision (M4Raw, 0.3 T)

Plan, gates and rationale: [`MIDL2027/proposal_repetitions/proposal.md`](../../MIDL2027/proposal_repetitions/proposal.md).

| File | What it does |
|---|---|
| `m4raw.py` | Range-fetch single h5 files from the stored Zenodo zips; register repetitions (complex cross-power fit: sub-pixel shift + global phase); Ψ from repetition differences; ESPIRiT (crop 0); noise-corrected metrics, ENA and common-reference comparison; training slices with white noise |
| `train_reps.py` | Repetition score matching (Noise2Noise DSM with σ_eff ≥ floor, noise equalization), EMA fine-tuning, unbiased denoising comparison, two-regime / stop-at-floor samplers |
| `pilot.py` | P0 diagnostics, P1 training set, P3 resumable evaluation (JSONL), κ/λ selection, gate report |
| `storage.py` | Drive via `drive.mount`, falling back to the Drive API (VS Code Colab extension) |
| `build_notebook.py` → `pilot_colab.ipynb` | The Colab pilot. Run the copy `pilot_colab_run.ipynb` (gitignored), because rebuilding overwrites outputs |

Reuses `experiments/midl27/fdc.py` (SENSE operator, IK-DC sampler, FA-KGD spectral DC) and the Ambient-DPS 3 T brain EDM.

## Checked locally (CPU, val subject 2022061203)
- **Registration:** synthetic shifts recovered to 1e-4 px.
- **Whitening:** Ψ from repetition differences gives a whitened difference variance of 1.000, flat across k-space.
- **Estimator:** a fully sampled single average scores ENA 1.05–1.08 (target 1). Thermal-only correction gives 0.89; the physiological variance between repetitions is about 12–21% of the thermal variance. The 2-average check gives 1.6–1.8 (target 2), so gates use the exact common-reference comparison.
- **Coil maps:** ESPIRiT crop 0.95 (the fastMRI default) leaves 35% of the coil signal unexplained at this SNR; crop 0 leaves 10%. With crop 0.95, R=2 collapsed to about 15 dB for every method.
- **Zero-shot 3 T prior** at R=2 (one slice, T=30): 23.0 dB, ENA 1.33. It already beats the fully sampled 1-average (common reference), but not the 2-average. That's why the GO gate is the 2-average. At R=4 it scores 18.8 dB, ENA 0.5.
- **End-to-end smoke run:** the P0/P1 cache, 2 fine-tuning steps with resume, denoising evaluation, P3 with all methods, JSONL resume, gate report.

Not run locally: the Colab-only parts (storage, a full training run on the A100).
