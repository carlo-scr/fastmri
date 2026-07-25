# PVAS WACV 2027 Paper Plan (single source of truth)

Target: WACV 2027 Round 2 (submit 2026-08-28 AoE, no rebuttal). 8 pages incl. figures/tables,
references extra. Algorithms track. Double-blind. Template: official WACV author kit
(placeholder: WACV-2023 kit style files in paper/pvas_v2/; swap for the 2027 kit at submission).

## Title (working)
"One Trajectory Is Enough: Ensemble-Free Active Sensing with Diffusion Posterior Variance"
(alt: "PVAS: Posterior-Variance-Guided Active Sensing for Training-Free Diffusion Reconstruction")

## Claims (must match docs/lit_review.md — forbidden claims listed there)
- C1 (core): First ensemble-free active measurement selection *inside a single reverse-diffusion
  trajectory*: per-coordinate posterior variance diag Cov[x0|xt] from Tweedie's second-moment
  identity via ONE Hutchinson probe with a central finite-difference JVP (2 extra denoiser calls
  per refresh), smoothed in a domain-appropriate basis, used to score candidate measurements
  mid-trajectory. All five prior training-free active-sensing systems (AdaSense, ADS, DAL,
  Iollo BED-MRI, GSC) are ensemble-based and select between complete sampling runs.
- C2: One statistic, two uses. The same variance map is (i) a spatially resolved Kalman gain
  K_i = P_i/(P_i + sigma_i^2) for data consistency (estimator ADOPTED from the Tweedie-covariance
  literature: Manor&Michaeli ICLR'24, TMPD TMLR'24, CA-DPS, Free Hunch ICLR'25 — REPURPOSED here),
  and (ii) the acquisition score. No training, no policy network, no ensembles.
- C3: Matched-budget AND matched-NFE protocol on two CV domains: (D1) multi-coil MRI k-space line
  acquisition (fastMRI brain), (D2) block-pixel acquisition on natural images (CIFAR-10; CelebA-HQ
  qualitative). Ensemble-based selection (ADS/AdaSense-style particle variance) run head-to-head
  as a baseline at matched budget, with its NFE cost reported.
- Honest notes: PV-gate-only gains are small on static masks (report as-is); adaptivity gains
  concentrate at high acceleration / low budgets (report where they vanish, cite Bakker MIDL'22).

## Notation (all sections MUST use this)
- Image x0 in R^d or C^d; forward operator A_M x0 + n = y, with M an index set of acquired
  "measurement atoms" (k-space lines / pixel blocks); n ~ N(0, diag(sigma_i^2)).
- Diffusion prior: VP/DDPM or EDM; denoiser mu_theta(x_t, t) approx E[x0|x_t];
  effective noise level sigma_t (EDM) or sigma_t^2 = (1-abar_t)/abar_t (VP, on xtilde = x_t/sqrt(abar_t)).
- Tweedie second moment: Cov[x0|x_t] = sigma_t^2 * (d mu_theta / d xtilde).
- PV estimate: P_hat = sigma_t^2 * (Jv ⊙ v) (unbiased for diagonal), v ~ N(0,I),
  Jv approx [mu(x_t + eps*sqrt(abar_t)*v) - mu(x_t - eps*sqrt(abar_t)*v)] / (2*eps);
  MRI variant uses |F(Jv)|^2 (rank-1, positive) + log-domain radial polynomial smoothing;
  natural-image variant uses channel-mean + Gaussian spatial smoothing + block pooling.
- Gate: K_i(t) = P_i(t) / (P_i(t) + sigma_i^2). Acquisition score of atom a: s(a) = sum_{i in a} P_i(t)
  (optionally / sigma_i^2). Rounds R_a, budget L atoms, window [t_a, t_b] late in the trajectory.

## Section plan (8 pages)
1. Introduction (~1 p). Hook: training-free diffusion solvers are passive; ensembles are the
   only known way to get acquisition-grade uncertainty, and they cost 16-256 trajectories.
   Fig. 1: teaser — PV map evolution + acquired atoms on both domains, NFE cost bar.
2. Related work (~0.75 p). Three axes: (i) training-free diffusion inverse solvers,
   (ii) posterior covariance in diffusion (adopted), (iii) active acquisition: trained policies
   (LOUPE/Bakker/Pineda/SeqMRI/A-DPS/Zhang) vs training-free ensemble-based (AdaSense/ADS/DAL/
   Iollo/GSC/van-de-Camp-ICASSP'23) vs ours. Positioning table (compact) from docs/lit_review.md §2.
3. Method (~2 p). 3.1 general formulation (operator-agnostic); 3.2 PV estimator + bias/cost
   analysis + analytic Gaussian validation (Fig. 2); 3.3 gated DC; 3.4 active selection +
   budget/NFE accounting; Algorithm 1 (general); 3.5 per-domain instantiation D1 (radial
   smoothing, SENSE multi-coil) and D2 (spatial smoothing, block pooling).
4. Experiments (~3 p).
   4.1 Setup: protocol rules (matched budget, matched NFE, paired Wilcoxon, seeds).
   4.2 D2 CIFAR: main table (6 policies incl. ensemble-K at matched budget; NFE column),
       calibration experiment (Spearman vs oracle; PV-1probe vs PV-4probe vs ensemble-K),
       gate x active 2x2 ablation, budget sweep curve. CelebA-HQ 256 qualitative.
   4.3 D1 MRI: main table (R in {4,8}, L in {8,16}; pigdm_mc / pv / pv_active / random_adaptive /
       static_match / oracle_active), NFE-matched comparison, PV-map evolution figure.
   4.4 Provenance-tagged comparison to published numbers of ADS/AdaSense/Iollo [CITED].
5. Discussion & limitations (~0.5 p). Estimator variance; ensemble baselines at full scale
   pending GPU; where adaptivity does not help; no clinical claims; retrospective protocol.
6. Conclusion (short).

## Results provenance tags (used in ALL tables/figures)
- [R-LOCAL]  = reproduced by code in this repo, run on Apple M4 Pro (MPS/CPU), seeds recorded.
- [CITED]    = numbers from the cited paper, never re-run by us.
- [PENDING]  = experiment specified + notebook prepared but not yet run (no numbers shown;
               table cell shows "—"). NO hypothesized numbers anywhere.

## Experiments inventory (status at plan time)
- Analytic Gaussian estimator validation: DONE [R-LOCAL] (poc_natural/sanity_gaussian.py):
  ratio 0.994-1.005 across t in {100..900}, corr 0.86-0.97 with 8 probes, no smoothing.
- CIFAR main grid: poc_natural/run_cifar.py — RUNNING.
- CIFAR calibration: poc_natural/calibration.py — QUEUED.
- MRI local grid 30 slices x {R4/R8} x {L8/L16}: scripts/run_pvas_local_grid.sh — RUNNING.
- MRI 1-slice reproduction: DONE [R-LOCAL]: pv_active 31.77 vs static 31.00 SENSE-PSNR (CPU),
  31.93 vs 31.25 (MPS).
- CelebA-HQ qualitative: QUEUED.
- Full-scale fastMRI (192 slices), knee cross-domain, ADS/AdaSense/Iollo head-to-head,
  LoDoPaB-CT: [PENDING] — Colab notebooks prepared, require A100.

## Legacy real numbers reusable from the CS231N project [R-LOCAL, 192 slices, brain AXFLAIR]
- PiGDM vs FA-KGD T-scaling (oracle-init): R4 +0.111..+0.505 dB (T10->T100); R8 +0.047..+0.249.
- VarNet supervised reference: R4 34.60 PSNR / 0.856 SSIM; R8 33.30 / 0.830.
- These belong to the FA-KGD paper; cite as "our prior work" ONLY if de-anonymization safe —
  default: fold FPDC/Kalman-DC lineage into method section without self-citation.
