# Repetitions as supervision: post-training diffusion priors for low-field MRI

**Status:** proposal plus pilot, drafted 2026-10-09. Planning document, not paper text: MIDL 2027 allows LLMs only for minor language edits, so the authors write the submission prose.
**Target:** MIDL 2027 full paper (abstract 2026-11-30, paper 2026-12-04).
**Replaces:** FA-KGD as the headline. G1 gave +0.3 dB on IK-DC only; ΠGDM was broken, and step degradation was not reproduced under per-T tuning. FA-KGD stays as an optional low-SNR component.
**Pilot code:** `experiments/midl27_reps/` (notebook `pilot_colab.ipynb`).

---

## 1. Claim (to be confirmed by the pilot)

Low-field MRI compensates for low SNR by acquiring several averages and summing them. Each repetition carries **independent noise**. That is exactly the supervision needed to learn a *clean-image* diffusion prior without clean data. We post-train a 3 T diffusion prior on the routine repetitions of a 0.3 T scanner. We reconstruct from **one undersampled average**, and we measure the result in **scan time**: the number of fully sampled averages that the reconstruction matches (equivalent number of averages, ENA).

Target headline, if the numbers support it: *one average at R=2 matches a 2-average (or better) fully sampled scan, i.e. ≥4× shorter, with one prior that also serves unseen accelerations, post-trained from N subjects' routine repetitions.*

## 2. Why it works: method in four lines

- Repetitions x_r = x + n_r and x_s = x + n_s (complex, coil-combined, prewhitened, registered).
- Train D(x_r + σz; σ_eff) → x_s. The optimum is E[x_s | input] = **E[x | input]**, because n_s is independent of the input (Noise2Noise applied to denoising score matching).
- The input already carries n_r, so condition on σ_eff = √(floor² + σ²). The model is defined only for **σ_eff ≥ floor**.
- **Noise equalization:** add complementary noise so the total noise is uniform at the floor, including the background and the never-acquired phase-encode columns. σ_eff is then exact at every pixel.
- **Below the floor** the data cannot teach the prior anything. Either stop at the floor and output the DC-corrected Tweedie estimate (`ft_stop`), or hand over to the frozen 3 T prior (`ft_two`). This is a designed ablation.

## 3. Data facts (verified 2026-10-09, M4Raw Zenodo 8056074)

| | train | val | test |
|---|---|---|---|
| subjects | 128 | 30 | 25 |
| repetitions T1 / T2 / FLAIR | 3 / 3 / 2 | 3 / 3 / 2 | 6 / 6 / 4 |

- The zips are **stored** (uncompressed), so single h5 files (~12 MB) can be range-fetched. No 12.7 GB download is needed, on Colab or on the Mac.
- k-space is (18 slices, 4 coils, 256, 256), FOV 240 mm (0.94 mm pixels). There is **no readout oversampling**. About 196 of 256 phase-encode columns are acquired (symmetric zero-padding).
- **SNR ≈ 8 per coil** (median brain / noise std): noise is a large part of the image.
- **Coil noise correlation** reaches up to 0.25. Ψ is estimated from **differences between repetitions**: after whitening, the variance of the outer k-space difference is 0.999–1.000, and it is flat across k-space (no receiver filter roll-off).
- **Drift between repetitions:** global phase moves 16–66°, and the shift is up to 0.33 px, mainly along phase-encode. A complex cross-power fit recovers synthetic shifts to 1e-4 px.
- **Coil maps:** ESPIRiT with the fastMRI eigenvalue crop (0.95) zeroes about 10% of the head at this SNR and leaves 35% of the coil signal unexplained. That made every method, classical included, collapse at R=2. With crop 0 the unexplained part is 10%, and R=2 unfolding is well conditioned (median condition number 2.5 with 4 coils).
- **Zero-shot 3 T prior (one slice, illustrative only):** at R=2 it reaches 23.0 dB corrected PSNR, ENA 1.33. It already beats a fully sampled single average on the common reference, but not the 2-average. At R=4 it scores 18.8 dB, ENA 0.5, with 4 coils. FA-KGD adds +0.1 dB at R=2.
- **Noise floor** σ ≈ 0.05–0.07 (normalized, per real channel), so about **25%** of T=50 sampling steps fall below it (kill threshold 30%).
- **Physiological variation between repetitions** is ≈ **11–21% of the thermal noise variance** in the coil-combined image. It is signal-dependent (bright regions +9.5%) and survives rigid registration. Consequences:
  - an evaluation that ignores it is biased: a fully sampled single average scores ENA 0.89 instead of 1;
  - the Noise2Noise target is still valid if the variation is zero-mean, but it sets a floor on achievable error. This is worth reporting in its own right.

## 4. Evaluation protocol (no ground truth, still unbiased)

The reconstruction uses **repetition 0 only**: its coil maps, scale and measurements. The held-out repetitions are therefore independent of it.

1. **Corrected error and ENA.** E|x̂ − x̄_held|² = |x̂ − x|² + v_rep / n_held, where v_rep = thermal + physiological variance per repetition, estimated from repetition pairs: E|x_r − x_s|²/2. ENA = v_rep / corrected MSE. On real data, a fully sampled single average scores 1.04–1.08 (target 1). The 2-average check gives 1.6–1.8 (target 2), so the model-based ENA is approximate. This is likely due to time-correlated drift; re-check on the 6-repetition test split.
2. **Common-reference comparison** (exact, used for the gates). Score the reconstruction, the fully sampled 1-average and the fully sampled 2-average against the **same** last repetition. The noise term is identical, so differences are unbiased. "Beats a k-average scan" is decided here.
3. **Secondary:** magnitude SSIM against the held-out average (biased; descriptive only).
4. The unit of analysis is the subject. Use paired differences with bootstrap CIs and Wilcoxon tests, as in the existing `analysis.py`.

## 5. Baselines and ablations (full study)

| Baseline | Question it answers |
|---|---|
| Zero-shot 3 T prior (native grid; **plus resampled to the prior's pixel size** if the gap is large) | Is post-training needed? (pixel-size fairness) |
| Single-scan self-supervision: SURE-score / **MSM** (ICLR 2026, public code) | Do repetitions beat single noisy scans? |
| **VarNet trained with repetition targets** (Noise2Noise, non-generative) | Is a generative prior needed? Expect a tie in-distribution and a diffusion win at unseen R or masks. |
| Same network trained from scratch on M4Raw | Does 3 T pretraining matter (the "post-training" claim)? |
| ESPIRiT + L1-wavelet; fully sampled K-averages | Classical anchor; the averaging curve |
| Nila-style zero-shot noise adaptation | Closest concurrent zero-shot approach |

Ablations: number of training subjects (10/30/60/128, the label efficiency for a new site); LoRA vs full fine-tuning; noise equalization on/off; `ft_stop` vs `ft_two`; unseen contrast (train T1+T2, test FLAIR); unseen R (6) and random masks; FA-KGD spectral DC (its theory predicts gains at exactly this SNR).

**Figure 1:** corrected error against acquisition time. It shows the fully sampled averaging curve (1–6 averages, test split) and our points (one average at R = 2, 3, 4).

## 6. Expected gains (hypotheses, not results)

| Comparison | Expected | Basis |
|---|---|---|
| Post-trained vs zero-shot 3 T | +1 to +3 dB | Typical MRI domain-shift losses; AdaDiff-style adaptation recovers about 1–2 dB; 0.3 T adds contrast, pixel-size and slice-thickness shift |
| vs single-scan self-supervision | +0.5 to +1.5 dB | Unbiased repetition targets versus SURE-type estimator variance |
| vs repetition-trained VarNet | tie at training R; win at unseen R/masks | End-to-end models usually win in-distribution |
| ENA at R=2 | 2–4 (zero-shot ≈ 1.3) | Learned denoisers at 0.3 T; capped by the physiological floor (v_rep/δ² ≈ 6–9) |

## 7. Pilot (decides go/no-go; about 3 GPU-hours plus fetching)

| Step | What | Output | Gate (fixed before running) |
|---|---|---|---|
| P0 | Diagnostics on 10 val subjects × 3 contrasts | floor, steps below floor, physiological fraction, registration, estimator check | **Kill** if median steps below floor > 0.30, or median physiological fraction > 0.5, or the 1-average estimator check falls outside [0.9, 1.2] |
| P1 | Training set: 40 train subjects × 3 contrasts × 8 slices (~960 slices, ~3.8 GB range-fetched) | cached npz on Drive | — |
| P2 | Full fine-tune, about 6k steps (A100), then the **unbiased denoising comparison** against the 3 T prior on held-out val pairs at σ ∈ {floor, 0.1, 0.2, 0.5, 1} | denoising MSE per σ | **Continue** if post-training lowers the N2N MSE at the floor by ≥ 10% |
| P3 | Reconstruction: κ ∈ {0.3, 1, 3} tuned on 4 val subjects, evaluated on 12 others × 3 contrasts × 4 slices, R ∈ {2, 4}, T=50; zero-shot vs `ft_stop` vs `ft_two` vs L1-wavelet | JSONL, gate table | **GO** if the best variant is ≥ +1 dB over zero-shot (subject × contrast CI excluding 0) at both R **and** beats the fully sampled **2-average** at R=2 (common reference) in ≥ 50% of units. **Strong** at ≥ 80%. (Zero-shot already beats the 1-average at R=2, so that bar is uninformative.) |

If the pilot fails, fall back to option A/B from the 2026-10-09 discussion (DC-ingredient decomposition, lesion-level calibration).

## 8. Timeline to 4 December

| Dates | Work |
|---|---|
| Oct 10–13 | Pilot P0–P3; go/no-go **Oct 14** |
| Oct 14–26 | Full training set (128 subjects); LoRA vs full; floor handling; MSM and SURE baselines; N2N-VarNet training |
| Oct 27–Nov 9 | Test split: averaging curve (6 repetitions), main table, unseen R/contrast, label efficiency |
| Nov 10–16 | Figures, statistics, robustness; freeze numbers |
| Nov 17–30 | Authors write the paper; abstract registration by Nov 30 |
| Dec 1–4 | Polish, supplement, submit |

## 9. Risks

1. **Physiological variation caps the gains.** If v_rep/δ² is small, ENA saturates. Measured ≈ 6–9× headroom on one subject; confirm in P0.
2. **Pixel-size mismatch** (0.94 mm vs fastMRI ≈ 0.7 mm) handicaps the zero-shot baseline. Add a resampled zero-shot before claiming a gap.
3. **Noisy targets.** Noise2Noise training converges more slowly; P2 measures whether the denoiser improves at all.
4. **Concurrent work** (MSM, Nila, AdaDiff). Our distinction: repetition supervision of a diffusion prior, cross-field post-training, and evaluation on raw data in scan-time units. Run an arXiv check before submission.
5. **Limitations to state:** healthy volunteers only, one vendor (XGY 0.3 T), retrospective undersampling, no reader study. Artifacts that repeat across repetitions are learned as signal.

## 10. Key references

- M4Raw (Sci Data 2023), https://zenodo.org/records/8056074
- Noise2Noise (Lehtinen et al., ICML 2018); Noise2Average, https://pmc.ncbi.nlm.nih.gov/articles/PMC13015446/
- MSM, Measurement score-based diffusion (ICLR 2026), https://arxiv.org/abs/2505.11853
- Nila, noise-level adaptive diffusion priors for low-field MRI (NMR Biomed 2026), https://pmc.ncbi.nlm.nih.gov/articles/PMC13475394/
- AdaDiff (MedIA 2023), https://arxiv.org/abs/2207.05876
- Ambient-DPS brain prior (Aali et al.); EDM (Karras et al., NeurIPS 2022)
- Kellman & McVeigh 2005 (SNR units, prewhitening)
