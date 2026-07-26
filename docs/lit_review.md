# PVAS Literature Review Synthesis (WACV 2027 target)

> **Superseded on 2026-07-25.** This file preserves the pre-audit search
> notes, but its proposed novelty sentence and several “matched NFE”
> statements are no longer valid. ADS already performs mid-trajectory active
> diffusion, and newer 2026 work further narrows the gap. Use
> [`sketchsense_proposal.md`](sketchsense_proposal.md) for the active claim,
> related-work positioning, baselines, and go/no-go criteria. Use
> [`protocol_audit.md`](protocol_audit.md) for the invalidated-result record.

Synthesized 2026-07-07 from an 8-lens verified literature sweep (106 papers screened, 18 verified-closest),
followed by an adversarial completeness pass (Section 0). Verified BibTeX for the 18 closest works:
`docs/verified_bibliography.json`.

---

## 0. Completeness-pass addenda (adversarial second sweep) and FINAL DECISION

**Final judgement: proceed-with-changed-claims.** The core wedge survives — no found work drives
*mid-trajectory* acquisition from an *ensemble-free, single-trajectory* variance estimate, and none reuses
one variance map for both Kalman-gated data consistency and selection under matched NFE. But the field is
more crowded than the first sweep assumed. There are at least FIVE training-free diffusion active-sensing
systems (AdaSense ECCV'24; ADS TMLR'25; Diffusion Active Learning arXiv'25; Iollo et al. BED-MRI arXiv'25;
Generative Spectrum Cartography arXiv'25), spanning MRI, CT, ultrasound, NMR, and spectrum maps. Consequences:

1. "General across modalities" is **dropped as a novelty claim** — reframed purely as evaluation breadth.
2. All novelty language is scoped to the **efficiency/mechanism axis**: single trajectory, ~2 extra denoiser
   calls per refresh, mid-trajectory selection, one variance map serving both DC gating and acquisition.
   Every competitor found is ensemble-based and selects between complete sampling runs.
3. New must-cite works found by the completeness pass (all landing pages verified):
   - **Iollo, Oudoumanessah, Lartizien, Dojat, Forbes** — "Active MRI Acquisition with Diffusion Guided
     Bayesian Experimental Design" — arXiv:2506.16237 (2025). Training-free sequential BED/EIG for k-space
     with a diffusion prior; needs an N+M ensemble of trajectories; selects between complete runs; already
     benchmarks against (and slightly beats) ADS. Joins ADS/AdaSense in the obligatory head-to-head set.
   - **Pandey/Gu et al.** — "Generative Spectrum Cartography: Unified Reconstruction and Active Sensing via
     Diffusion Models" — arXiv:2512.20108 (2025). Closest *framing* competitor ("unified reconstruction +
     active sensing"); ensemble-based uncertainty, between-run selection, single modality (radio maps).
   - **van de Camp, Joudeh, Antunes, van Sloun** — "Active Subsampling Using Deep Generative Models by
     Maximizing Expected Information Gain" — ICASSP 2023 (IEEE 10097104). EIG-with-generative-posteriors
     predates ADS/AdaSense; must be cited as lineage.
   - **Goffinet et al.** — DiffNMR2 (arXiv:2502.05230, 2025): diffusion-uncertainty-guided NMR sampling.
   - **van Nierop, Nolan, Stevens, van Sloun** — Patient-adaptive cognitive ultrasound (arXiv:2508.08782).
   - **Ayzenberg et al.** — token-uncertainty active MRI (arXiv:2603.21806, 2026): independent evidence that
     "project uncertainty into measurement space" is common ground, not ours.
   - **Ravula et al.** — diffusion-driven *non-adaptive* mask optimization (arXiv:2306.03284): natural
     "diffusion, non-adaptive" control arm for matched-budget tables.

**Chosen claims for the paper (post-review):**
- C1 (core): first *ensemble-free* active measurement selection inside a single reverse-diffusion trajectory,
  driven by a Tweedie second-moment posterior-variance estimate at O(1) denoiser-call overhead per refresh.
- C2: one statistic, two uses — the same variance map serves as a spatially resolved Kalman gain for data
  consistency (adopted estimator, repurposed) and as the acquisition score; neither requires training,
  a policy network, or posterior ensembles.
- C3: an honest matched-budget AND matched-NFE evaluation protocol, instantiated on two CV domains
  (multi-coil MRI k-space lines; pixel-block acquisition on natural images), with ensemble-based selection
  (ADS/AdaSense-style) run as a baseline at matched budget and matched NFE.
- Forbidden claims remain forbidden (see §1).

---

## 1. Verdict

**The contribution is novel only in its specific combination, not in any of its individual parts — and the paper lives or dies on wording and matched-NFE evidence.** Every constituent idea is already published: training-free active acquisition with a pretrained diffusion prior exists three times over (AdaSense ECCV 2024; ADS TMLR 2025; Diffusion Active Learning 2025), "score candidate measurements by projected posterior variance" is literally AdaSense's Eq. 5-6 and Diffusion Active Learning's CT criterion, the variance-gated Kalman-style data-consistency gain is anticipated in structure by PiGDM/DDRM and in substance by Peng et al. (ICML 2024), TMPD (TMLR 2024), CA-DPS (arXiv 2024), and Free Hunch (ICLR 2025), and single-trajectory Tweedie-second-moment/Jacobian variance estimation itself appears in Manor & Michaeli (ICLR 2024), CA-DPS, Free Hunch, Schioppa & Salimans (2026), and Divergence-is-Uncertainty (2026). **What no paper does — the defensible gap — is this exact sentence:** *"PVAS is the first method to drive mid-trajectory active measurement selection from an ensemble-free, single-trajectory posterior-variance estimate (Tweedie second moment via one Hutchinson finite-difference JVP, ~2 extra denoiser calls per step), reusing the same variance map as a Kalman-gated data-consistency gain, and evaluated under matched NFE and matched measurement budget across MRI, CT, and compressive-sensing operators."* Every prior active-sensing method pays for uncertainty with an ensemble (16 particles in ADS; s full posterior-sampling rounds in AdaSense; posterior-sample ensembles in Diffusion Active Learning / Iollo), and no covariance-estimation paper does acquisition. Forbidden claims: "first training-free active acquisition," "first posterior-variance-based selection," "first Tweedie-covariance estimator," "novel Kalman-gated data consistency." Contribution (a) must be framed as an enabling component of (b) unless it empirically beats Peng/CA-DPS/Free Hunch/TMPD at matched NFE.

---

## 2. Positioning Table

| Method | Needs training / policy net? | Needs posterior ensemble? | Adaptive at inference? | Variance / uncertainty source | Domains shown | Venue + Year |
|---|---|---|---|---|---|---|
| **AdaSense** (Elata et al.) | No (zero-shot; only unconditional prior) | **Yes** — s full DDRM posterior-sampling runs per round (e.g. 256 trajectories/image for faces) | Yes — between complete sampling rounds | Empirical PCA/covariance of posterior-sample ensemble | Faces (CS), fastMRI knee lines, DeepLesion CT angles | ECCV 2024 |
| **ADS** (Nolan et al.) | No | **Yes** — Np=16 parallel DPS particles (16× NFE/step) | Yes — mid-trajectory, on a schedule | GMM entropy of particle Tweedie means in measurement space | MNIST px, CelebA boxes, fastMRI lines, ultrasound (CAMUS) | TMLR 2025 |
| **Diffusion Active Learning** (Barba et al.) [UNVERIFIED metadata] | No | **Yes** — ensemble of conditional posterior samples | Yes — sequential CT angle rounds | Projected posterior variance across sample ensemble | CT angles | arXiv 2504.03491, 2025 |
| **Iollo et al.** (diffusion BED) [UNVERIFIED metadata] | No | **Yes** — multi-sample Monte-Carlo EIG | Yes — sequential design | EIG over posterior-sample sets | MRI (and BED benchmarks) | ICLR 2025 / arXiv 2025 |
| **Zhang et al.** (active MRI, evaluator) | **Yes** — jointly trained reconstructor + adversarial evaluator | No | Yes — greedy outer loop | Learned heteroscedastic variance head (used to halt; selection via evaluator score) | MRI (simulated k-space), grayscale ImageNet | CVPR 2019 |
| **Pineda et al.** (SS/DS-DDQN) | **Yes** — offline-RL DDQN policy + trained CNN reconstructor | No | Yes (ss-ddqn) / No (ds-ddqn fixed trajectory) | None — learned Q-values | fastMRI single-coil knee | MICCAI 2020 |
| **Bakker et al.** (greedy policy search) | **Yes** — policy-gradient CNN + fixed U-Net | No (q=8 rollouts at training only) | Yes — per-image sequential | None — learned softmax policy | fastMRI knee + brain | NeurIPS 2020 |
| **Bakker et al.** (multi-coil policies) | **Yes** — policy + E2E VarNet, jointly trained | No | Nominally (one-shot from ACS); learned policies converge to **non-adaptive** | None — learned saliency heatmap | fastMRI multi-coil knee | MIDL 2022 |
| **LOUPE / A-DPS / SeqMRI** (learned masks & policies) | **Yes** — end-to-end trained masks / LSTM policy / sampler+reconstructor | No | LOUPE: no (fixed mask); A-DPS & SeqMRI: yes via trained policy | None — learned logits/heatmaps | MRI (LOUPE, A-DPS, SeqMRI); MNIST/CIFAR (DPS/A-DPS) | IPMI 2019 / ICML 2021 / ML4H 2021 |
| **PiGDM** (Song et al.) | No | No — single trajectory | No — fixed operator | Hand-crafted **scalar** r_t² = σ_t²/(σ_t²+1), same for all coords | SR, inpainting, JPEG, deblur (ImageNet) | ICLR 2023 |
| **Peng et al.** (optimal posterior covariance) | Partially — best variant trains a covariance head; "Convert" needs iDDPM variance head; "Analytic" needs offline MC | No (at inference) | No — fixed operator | Model reverse-variance head converted to diag Cov[x0\|xt], or offline MC, or trained DCT/DWT head | Inpainting, deblur, SR (FFHQ/ImageNet) | ICML 2024 |
| **TMPD** (Boys et al.) | No | No — single trajectory | No — fixed operator | Tweedie 2nd moment: autodiff VJP row-sum/diagonal of denoiser Jacobian | GMMs, inpainting, SR (natural images) | TMLR 2024 |
| **CA-DPS** (Hamidi & Yang) | No | No — single trajectory | No — fixed operator | Tweedie 2nd moment via **temporal** finite difference of scores across timesteps (~0 extra NFE) | Inpainting, deblur, SR (FFHQ/ImageNet) | arXiv 2412.20045, 2024 (unpublished) |
| **Free Hunch** (Rissanen et al.) | No | No — single trajectory | No — fixed operator | DCT-diagonal + low-rank BFGS secant updates from existing denoiser calls (**0 extra NFE**) | Deblur, inpainting, SR (ImageNet), GMMs | ICLR 2025 |
| **Schioppa & Salimans** | No | No — single trajectory | No — unconditional sampling only | Tweedie Jacobian via 1 autodiff JVP + Hutchinson, frequency-block-diagonal | Unconditional generation (ImageNet, CIFAR) — **no inverse problems** | arXiv 2605.13910, 2026 |
| **PVAS (ours)** | **No** — no training, no policy, any pretrained prior | **No** — single trajectory + 1 Hutchinson central-FD JVP (2 extra denoiser calls/step, no autodiff) | **Yes — mid-trajectory measurement selection** | Tweedie 2nd moment: σ_t² · diag(denoiser Jacobian), Hutchinson probe, radially smoothed in k-space | MRI k-space lines, CT angles, inpainting/CS (planned) | WACV 2027 (target, R2 deadline 2026-08-28) |

---

## 3. Five Biggest Threats to Novelty (brutally honest)

### Threat 1 — CA-DPS (Hamidi & Yang, arXiv:2412.20045): **kills contribution (a) as a standalone claim.**
They already state Tweedie's second-moment identity as a theorem, extract a training-free diagonal Cov[x0|xt] from a single trajectory, and plug it into a PiGDM-style likelihood N(A x̂0, σ²I + AΣAᵀ) — which for subsampled Fourier operators **is** our per-frequency gain P_i/(P_i+σ²). We cannot claim "single-trajectory Tweedie-based variance-gated data consistency" as new. **Differentiating argument (narrow but real):** their estimator is a temporal finite difference of scores across consecutive timesteps, conflating change-in-x with change-in-t — a heuristic; ours is a Hutchinson central-difference JVP at fixed t, unbiased for the diagonal in expectation, plus domain-aware radial k-space smoothing, plus they have zero medical/multi-operator experiments and no acquisition. This delta only survives if we **empirically show** better variance calibration and reconstruction at matched NFE. Also note it appears unpublished — cite it anyway; reviewers will know it.

### Threat 2 — ADS (Nolan et al., TMLR 2025): **kills "first training-free / mid-trajectory / policy-free active acquisition."**
ADS already does measurement selection *inside a single reverse pass* of a pretrained prior, training-free, no policy network, interpretable entropy criterion, on fastMRI lines — matching supervised PG-MRI/LOUPE/SeqMRI. Our categorical claims are dead; only quantitative ones survive. **Differentiating argument:** ADS needs Np=16 particle trajectories (16× denoiser NFEs per step) to get uncertainty from particle disagreement; PVAS gets diag Cov[x0|xt] analytically from ONE trajectory at 2 extra calls/step (~3× vs ~16× per step), and additionally gates data consistency with the same variance map (ADS uses plain fixed-weight DPS). The headline must be "ensemble-free, O(1)-overhead" with a head-to-head matched-NFE table against ADS (their public code and ~40 ms/step figure make this feasible and obligatory).

### Threat 3 — AdaSense (Elata et al., ECCV 2024): **kills novelty of the selection criterion and of multi-domain generality.**
AdaSense's constrained selection rule is mathematically our rule — score each candidate row by empirical projected posterior variance — and it already spans CS/faces + MRI lines + CT angles and matches trained RL policies zero-shot. "Projected-posterior-variance active sensing across operators" is taken. **Differentiating argument:** AdaSense's variance comes from s complete posterior-sampling runs per acquisition round (hundreds of full trajectories per image; the authors concede the cost), restarted between rounds — never mid-trajectory; it assumes noiseless measurements with hard DDRM consistency. PVAS: one trajectory total, mid-trajectory acquisition, noise-calibrated Kalman DC, matched-NFE evaluation that directly exposes their compute. We must run AdaSense as a baseline at matched budget AND matched NFE.

### Threat 4 — Free Hunch (ICLR 2025) + Schioppa & Salimans (2026) + Divergence-is-Uncertainty (2026): **kills novelty of the estimator itself, and pressures our 2-extra-NFE cost from below.**
FH gets single-trajectory Cov[x0|xt] with **zero** extra NFEs (BFGS secant = finite-difference JVP along the trajectory's own steps, DCT-diagonal structure) and already uses it in exactly the (AΣAᵀ+σ²I)⁻¹ gain. Schioppa & Salimans already do Hutchinson-probe Tweedie-Jacobian variance with frequency-domain diagonal structure at ~1 extra NFE/step. Divergence-is-Uncertainty gives single-pass posterior covariance for flow matching. "Cheap single-trajectory Tweedie variance with frequency structure" is fully occupied. **Differentiating argument:** none of them selects measurements — the estimator must be presented as *adopted/adapted* (cite all three + Manor & Michaeli + TMPD), with our contribution being its repurposing as an acquisition signal. And we must justify +2 NFE/step against FH's 0 at matched total NFE — if FH's covariance drives selection equally well, our estimator contribution evaporates entirely; test this ablation ourselves before a reviewer does.

### Threat 5 — Diffusion Active Learning (arXiv:2504.03491) + Peng et al. (ICML 2024) + Bakker MIDL 2022: **kills CT-selection novelty, the "optimal covariance in guidance" framing, and — most dangerously — the premise that adaptivity helps.**
DAL already selects CT angles by *maximizing projected posterior variance* with a pretrained diffusion prior, training-free — so even our CT instantiation of the selection rule is anticipated (ensemble-based, again). Peng et al. formalize optimal diagonal posterior covariance in both guidance types; if we cannot beat their variants at matched NFE, contribution (a) must be demoted to an enabling component. And Bakker MIDL 2022 found the best *learned* policies converge to non-adaptive masks (adaptive ≈ non-adaptive at 4×) — a reviewer will wield this against the entire premise. **Differentiating argument:** run DAL as the CT baseline at matched NFE; cite Bakker honestly and show PVAS's wins concentrate where adaptivity plausibly matters (high acceleration, high-variance k-space regions, out-of-distribution anatomy), noting their negative result concerned one-shot ACS-conditioned policies entangled with a jointly trained reconstructor — a different regime from mid-trajectory posterior-variance selection. If our gains also vanish at moderate acceleration, report it.

---

## 4. Evaluation Requirements for WACV Reviewers

**Venue reality check:** WACV 2027 Round 1 is closed; the only path is Round 2 (registration 2026-08-21, submission 2026-08-28 AoE, supplementary 2026-08-30) with **no rebuttal** — accept/reject only. 8 pages incl. figures/tables, Algorithms track, official author-kit ZIP from wacv.thecvf.com. A Round-1-rejected paper resubmitted as "new" in R2 is desk-rejected (N/A for us). arXiv preprints of competitors are citable but cannot be grounds for rejection.

**Mandatory baselines (head-to-head, not just cited):**
1. **Active, training-free:** AdaSense (ECCV 2024, code public) and ADS (TMLR 2025, code at github.com/active-diffusion-subsampling/ads) — on MRI at minimum; Diffusion Active Learning if we claim CT.
2. **Active, trained-policy:** Pineda SS/DS-DDQN (facebookresearch/active-mri-acquisition harness) and/or Bakker 2020 — the community-standard foil for "no policy network."
3. **Passive solvers (for contribution a):** DPS, PiGDM, DDRM, plus at least two covariance-aware methods — Peng et al. (Convert/Analytic), TMPD (row-sum diagonal), CA-DPS, Free Hunch — at matched NFE. DDS (ICLR 2024) as the MRI/CT efficiency benchmark.
4. **WACV-venue calibration:** at least one WACV-accepted diffusion-MRI method (R3DM WACV 2025 or BGDM WACV 2025) to show we clear the venue's own bar.
5. **Non-adaptive mask baselines:** random, equispaced, low-frequency, variable-density, and an offline-optimized (LOUPE-style) mask at every budget — because Bakker/Pineda showed well-tuned fixed masks are hard to beat.

**Mandatory protocol elements:**
- **Matched-budget AND matched-NFE tables** — this is our entire selling point; report total denoiser calls per reconstructed image for every method (an NFE/cost table like ADS's 40 ms/step figure). Include a PVAS-at-equal-NFE vs ADS-at-16× comparison and a "give AdaSense our NFE budget" comparison.
- **Datasets:** fastMRI knee AND brain (single-coil for comparability with Pineda/ADS; multi-coil strengthens), volume-level statistics, multiple acceleration factors (4×, 8×, and a high-acceleration regime ≥16× where adaptivity should shine); SSIM/PSNR-vs-acceleration curves under fixed line budgets via the active-mri-acquisition harness. CT: LoDoPaB-CT (precomputed 1000-angle sinograms make angle-subsetting clean; Zenodo 10.5281/zenodo.3384092, CC BY 4.0). One non-medical operator (inpainting/CS on CelebA or CIFAR-scale) for the generality claim.
- **Estimator-quality experiment (load-bearing):** calibration of our Hutchinson diagonal vs ensemble ground-truth covariance, vs TMPD row-sum, Peng Convert, CA-DPS temporal-FD, and Free Hunch — showing selection quality matches ensemble entropy/variance at 5-10× fewer NFEs. Without this, the paper reads as "ADS with a cheaper estimator, unproven."
- **Ablations:** (i) Kalman DC on/off, (ii) active selection on/off (2×2 grid isolating each contribution), (iii) number of Hutchinson probes, (iv) radial smoothing on/off, (v) selection driven by FH-covariance instead of ours.
- **Honest negatives:** report where adaptive ≈ non-adaptive (cite Bakker MIDL 2022 proactively); avoid clinical-benefit over-claims.

---

## 5. Verified Bibliography

**Verified (metadata confirmed by the sweep):**

1. **Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling (AdaSense)** — Elata, Michaeli, Elad — ECCV 2024, Springer LNCS 15136 pp. 290-308 — https://arxiv.org/abs/2407.08256 (DOI 10.1007/978-3-031-73229-4_17)
2. **Active Diffusion Subsampling (ADS)** — Nolan, Stevens, van Nierop, van Sloun — TMLR 2025 — https://arxiv.org/abs/2406.14388 (OpenReview OGifiton47)
3. **Experimental design for MRI by greedy policy search** — Bakker, van Hoof, Welling — NeurIPS 2020 (spotlight) — https://arxiv.org/abs/2010.16262
4. **Active MR k-space Sampling with Reinforcement Learning** — Pineda, Basu, Romero, Calandra, Drozdzal — MICCAI 2020, LNCS 12262 pp. 23-33 — https://arxiv.org/abs/2007.10469 (DOI 10.1007/978-3-030-59713-9_3)
5. **Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance** — Peng, Zheng, Dai, Xiao, Li, Zou, Xiong — ICML 2024, PMLR 235 pp. 40347-40370 — https://arxiv.org/abs/2402.02149
6. **Pseudoinverse-Guided Diffusion Models for Inverse Problems (PiGDM)** — Song, Vahdat, Mardani, Kautz — ICLR 2023 — https://openreview.net/forum?id=9_gsMA8MRKQ (no arXiv version exists)
7. **Diffusion Posterior Sampling for General Noisy Inverse Problems (DPS)** — Chung, Kim, McCann, Klasky, Ye — ICLR 2023 (spotlight) — https://arxiv.org/abs/2209.14687
8. **Deep Probabilistic Subsampling for Task-Adaptive Compressed Sensing** — Huijben, Veeling, van Sloun — ICLR 2020 — https://openreview.net/forum?id=SJeq9JBFvH (no arXiv)
9. **Active Deep Probabilistic Subsampling (A-DPS)** — van Gorp, Huijben, Veeling, Pezzotti, van Sloun — ICML 2021, PMLR 139 pp. 10509-10518 — https://proceedings.mlr.press/v139/van-gorp21a.html (no arXiv)
10. **End-to-End Sequential Sampling and Reconstruction for MRI (SeqMRI)** — Yin, Wu, Sun, Dalca, Yue, Bouman — ML4H 2021, PMLR 158 pp. 261-281 — https://arxiv.org/abs/2105.06460
11. **Learning-based Optimization of the Under-sampling Pattern in MRI (LOUPE)** — Bahadir, Dalca, Sabuncu — IPMI 2019, LNCS 11492 pp. 780-792 — https://arxiv.org/abs/1901.01960; journal: IEEE TCI vol. 6 pp. 1139-1152, 2020 — https://arxiv.org/abs/1907.11374 (DOI 10.1109/TCI.2020.3006727)
12. **Denoising Diffusion Restoration Models (DDRM)** — Kawar, Elad, Ermon, Song — NeurIPS 2022 — https://arxiv.org/abs/2201.11793
13. **Enhancing Diffusion Models for Inverse Problems with Covariance-Aware Posterior Sampling (CA-DPS)** — Hamidi, Yang — arXiv preprint 2024 (no venue acceptance found as of 2026-07) — https://arxiv.org/abs/2412.20045
14. **Tweedie Moment Projected Diffusions for Inverse Problems (TMPD)** — Boys, Girolami, Pidstrigach, Reich, Mosca, Akyildiz — TMLR 2024 — https://arxiv.org/abs/2310.06721 (OpenReview hDzjO41IOO)
15. **Free Hunch: Denoiser Covariance Estimation for Diffusion Models Without Extra Costs** — Rissanen, Heinonen, Solin — ICLR 2025 — https://arxiv.org/abs/2410.11149 (OpenReview 4JK2XMGUc8)
16. **Covariance-aware sampling for Diffusion Models** — Schioppa, Salimans — arXiv preprint 2026 (no venue as of 2026-07) — https://arxiv.org/abs/2605.13910
17. **On the Posterior Distribution in Denoising: Application to Uncertainty Quantification** — Manor, Michaeli — ICLR 2024 — https://arxiv.org/abs/2309.13598 (OpenReview adSGeugiuj)
18. **Reducing Uncertainty in Undersampled MRI Reconstruction With Active Acquisition** — Zhang, Romero, Muckley, Vincent, Yang, Drozdzal — CVPR 2019 pp. 2049-2058 — https://arxiv.org/abs/1902.03051
19. **On learning adaptive acquisition policies for undersampled multi-coil MRI reconstruction** — Bakker, Muckley, Romero-Soriano, Drozdzal, Pineda — MIDL 2022, PMLR 172 pp. 63-85 — https://arxiv.org/abs/2203.16392

**Cited in lens assessments but NOT independently verified — confirm metadata before citing:**

20. [UNVERIFIED] **Diffusion Active Learning** (CT angle selection) — Barba et al. — arXiv:2504.03491, 2025 — check for v2 before submission
21. [UNVERIFIED] **Iollo et al., diffusion-guided sequential BED** — arXiv:2410.11826 (attributed to ICLR 2025 in one lens) and arXiv:2506.16237 (2025) — two IDs reported; resolve which is which
22. [UNVERIFIED] **Christoffel-DPS** (offline + online sensor selection for DPS, Adcock/Schönlieb groups) — arXiv:2605.06861, May 2026 — must read in full; likely to iterate
23. [UNVERIFIED] **Divergence is Uncertainty** (single-pass velocity-divergence posterior covariance for flow matching) — arXiv:2605.00941, May 2026
24. [UNVERIFIED] **DDS** — Chung et al., ICLR 2024 — arXiv:2303.05754 (efficiency benchmark for MRI/CT)
25. [UNVERIFIED] **STSL** — CVPR 2024 (Hutchinson trace probes in an inverse-problem sampler)
26. [UNVERIFIED] **Daras et al., survey of diffusion inverse-problem solvers**, 2024
27. [UNVERIFIED] **Jalal et al. 2021** (CS-MRI with score-based priors) and **Chung & Ye 2022, score-MRI** (ensemble-based UQ)
28. [UNVERIFIED] **UA-Flow** — arXiv:2602.10326 (Hutchinson diagonal Jacobian covariance in flow sampling, trained variance head)
29. [UNVERIFIED] **CDPA** — arXiv:2604.21960 (posterior-variance hallucination flags for CT)
30. [UNVERIFIED] **Gutha et al. WACV 2025** (MAP-reframed DC), **Askari et al. BGDM WACV 2025**, **Bangun et al. R3DM WACV 2025** — same-venue competitors/baselines
31. [UNVERIFIED] **Cognitive ultrasound / patient-adaptive echo** — van Sloun group, IEEE TUFFC 2024 / TMI 2026; **DiffATD**; trained-policy 2025 line (Du et al. arXiv:2505.04586, Xu & Oksuz STACOM 2025, Penninga et al. ICASSP 2025)
32. [UNVERIFIED] **FisherRF** (analytic information criteria for active view selection); **DAD** and **Contrastive-Diffusions BED** (BOED lineage); **Patel & Oberai** arXiv:2003.12597 (pre-diffusion generative-posterior-variance acquisition)

**Datasets (verified):** LoDoPaB-CT (Zenodo 10.5281/zenodo.3384092, CC BY 4.0, 42,895 slice pairs, 1000-angle sinograms); Mayo/AAPM LDCT-and-Projection-data (TCIA 10.7937/9npb-2637); CT-ORG (TCIA 10.7937/tcia.2019.tt7f4v7o); fastMRI knee/brain via facebookresearch/active-mri-acquisition harness.
