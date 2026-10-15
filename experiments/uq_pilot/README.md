# Exploration pilot: uncertainty, re-measurement, averaging, trust

A two-hour A100 notebook that checks which directions show signal on our data before we commit to one:

- **A**: does diffusion posterior spread predict the error?
- **B**: when to measure again (nested re-acquisition with a Learn-then-Test stop).
- **C**: when to measure longer (lines vs averages at 0.3 T; who gains from a second average).
- **D**: when not to trust (failure and OOD monitors under noise, motion and 3T → 0.3T).
- **E**: are small lesions lost, and is the loss silent?
- **F**: FA-KGD as uncertainty.
- **G**: where to measure.
- **H**: optional, the post-trained 0.3 T prior.

The literature anchors and the pre-set "promising" criteria are in the notebook's first two cells. The last cell prints a scoreboard.

| File | What it does |
|---|---|
| `uq.py` | Batched IK-DC posterior sampler (K samples + k-space hold-out members in one batch); exactly nested masks; per-column averaging weights; fastMRI and M4Raw cases; controlled shifts (noise, motion, synthetic lesions); per-slice summaries; decision analyses (cascade, LTT, allocation, coverage) |
| `build_notebook.py` → `uq_pilot_colab.ipynb` | The Colab notebook. It embeds `uq.py`, so it runs against the already-pushed branch. Run a copy (`*_run.ipynb` is gitignored) |
| `test_local.py` | Runs every non-Colab cell of the notebook on the CPU with T = 2 (smoke test only) |

Reuses `experiments/midl27/fdc.py` (SENSE, DC, spectrum) and `experiments/midl27_reps/m4raw.py` (fetch, registration, Ψ from repetitions, noise-corrected references).

## Run
1. In Colab, open an A100 runtime and open `uq_pilot_colab.ipynb`. Check that `network-snapshot.pkl` and `midl27_brain.tar` are in `MyDrive/fastmri_artifacts/`.
2. Run all. Cell 0.5 measures seconds per ensemble and shrinks the cohort to fit `BUDGET_MIN`. The full plan is 28 fastMRI volumes and 10 M4Raw subjects.
3. Results go to `fastmri_artifacts/uq_pilot_results/`: `records.jsonl`, figures, `scoreboard.csv` and `summary.json`. Every run cell resumes.

## Design notes
- **Errors on M4Raw** need no ground truth. The reconstruction uses repetition 0 only. The reference is the mean of the held-out repetitions, and its per-pixel variance is estimated from their spread and subtracted. That estimate is unbiased for T1/T2 (3 repetitions) and slice-level only for FLAIR (2 repetitions). The averaging probe (C) scores every configuration against repetition 2 only, so differences between configurations are unbiased.
- **Pairing:** within a slice, every variant (shift, spectral DC, lesion) reuses the reference run's mask, hold-out folds and sampler seed.
- **`pred`** = (1 + 1/K) × the sample variance. This is the predicted MSE of the K-sample mean if the samples were exact posterior draws, so the calibration ratio (actual / predicted) should be 1.
- **Not included:** fastMRI+ lesion boxes (E uses controlled synthetic lesions instead), the knee as a far-OOD set, and per-level thresholds in the cascade (one threshold on predicted NMSE, as in the Wen et al. stopping rule).
