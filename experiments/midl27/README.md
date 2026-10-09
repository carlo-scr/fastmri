# MIDL 2027 experiment suite (FA-KGD, frequency-resolved data consistency)

Fills the `XX.XX` cells in `MIDL2027/paper`. Plan and gates: `MIDL2027/revision_plan.tex`.

| File | What it is |
|---|---|
| `fdc.py` | The method and all baselines. One shared DC solver; "isotropic" vs "+FA-KGD" differ **only** in the per-frequency posterior variance `p_i(t)`. Hosts: IK-DC, ΠGDM, DDS, DiffPIR, DPS, plus the original CS231N sampler (`fakgd_v1_*`). |
| `run.py` | Data loading (prewhitening, ESPIRiT maps, MVUE target), resumable grid runner (JSONL), tuning and selection. |
| `analysis.py` | Volume-level PSNR/SSIM, paired bootstrap CIs, Wilcoxon, Holm, gates, LaTeX rows, figures. |
| `pack_data.py` | Packs N volumes × central slices into one tar for Drive. |
| `build_notebook.py` → `midl27_colab.ipynb` | The Colab experiment book, organised by the plan's gates. |
| `../../tests/test_midl27_fdc.py` | Solver vs closed-form Kalman and dense solves, mask budget, whitening, spectrum fit. |

## Quick start (Colab, A100)
1. Push this branch to GitHub (the notebook clones `BRANCH`).
2. Upload to `MyDrive/fastmri_artifacts/`:
   - `network-snapshot.pkl`, the supervised EDM (R=1) from Ambient-DPS (already used by older notebooks);
   - `midl27_brain.tar`: 28 AXFLAIR volumes × 8 central slices, 4.3 GB, already built at `data/midl27_brain.tar`. Rebuild with
     `python experiments/midl27/pack_data.py --src data/multicoil_val --acquisition AXFLAIR --n_volumes 28 --n_slices 8 --out data/midl27_brain.tar`.
3. Open `midl27_colab.ipynb` in Colab and run Stage 0 and then Stage 1. Stage 1 decides gate G1.

## Conventions (state these in the paper's setup)
- **Noise:** coil covariance Ψ is estimated from the extreme readout rows of the *acquired* columns. fastMRI zero-pads the phase-encode direction and ships no noise prescans. All k-space is prewhitened with Ψ, so the solver's likelihood is white with variance 1/scale².
- **Coil maps and target:** ESPIRiT (sigpy) on the central 24 fully sampled columns. The same maps are used for reconstruction and for the MVUE target. This is the convention of the prior's training data (Ambient-DPS) and of Jalal et al. 2021. With low-resolution maps from the R-dependent ACS, even a fully sampled SENSE combination scored only 21.8 dB against RSS. That mismatch capped every method, so the ESPIRiT/MVUE convention is required.
- **Masks:** equispaced with fully sampled centre (cf 0.08 at R=4, 0.04 at R=8). Zero-padded columns are excluded.
- **Normalisation:** 99th percentile of the RSS image computed from the acquired ACS only.
- **Metrics:** PSNR from stacked volume SSE with the volume maximum as data range, SSIM, NMSE, HFEN, on the central 320×320 crop of |MVUE|.

## Findings from local CPU checks (1 slice, NOT evidence)
- The original M-step is inert on real data: relative σ² change ≈1e-6 and identical PSNR to `m_step_mode=off`. This is plan C3.
- Exact Kalman DC with an isotropic prior variance as large as the diffusion noise (κ v_t) is ill-conditioned SENSE inversion. More CG iterations amplify aliasing (semi-convergence). Isotropic baselines therefore need κ / n_cg tuning; the grids in `run.py` cover this. The spectral variance is bounded by the image power spectrum, so it stays well conditioned.
- One-step denoising error (Tweedie + DC) is lower with the spectral variance than with the white (ΠGDM-style) variance at every σ tested.
- ΠGDM's guidance through the EDM Jacobian is 30–600× larger than the Kalman step at large σ, so it needs a small, tuned guidance weight (grid 1e-3…1).
- The Fourier preconditioner is exact single-coil but harmful with real coil maps, so it is off by default.

## Not implemented yet
- Ablation row (f), the noise-conditioned frequency exposure schedule of Tian et al. 2026.
- DAPS host.
- Knee: needs a knee prior (see plan C8) plus a `pack_data.py` run on knee multi-coil data.
- M4Raw: check that its HDF5 layout matches fastMRI (`kspace` as (slice, coil, H, W)), then pack it like the brain cohort.
- fastMRI+ ROI metrics (Tab. 6): need the fastMRI+ CSV and a box→crop mapping (note the vertical flip documented by fastMRI+).
