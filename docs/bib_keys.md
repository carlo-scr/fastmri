# Canonical BibTeX keys for the PVAS WACV paper

Section writers MUST cite using exactly these keys. The bibliography agent creates
paper/pvas_v2/pvas.bib with one verified entry per key (metadata from
docs/verified_bibliography.json where available, otherwise verified via web).
Any key that cannot be verified is listed in pvas.bib as a commented-out entry
with [UNVERIFIED] and must not be cited in the final draft.

## Active sensing / adaptive acquisition
- elata2024adasense — AdaSense, Elata/Michaeli/Elad, ECCV 2024, arXiv:2407.08256
- nolan2025ads — Active Diffusion Subsampling, Nolan et al., TMLR 2025, arXiv:2406.14388
- iollo2025bedmri — Diffusion-guided BED for active MRI, Iollo et al., arXiv:2506.16237
- barba2025dal — Diffusion Active Learning (CT angles), Barba et al., arXiv:2504.03491
- gu2025gsc — Generative Spectrum Cartography, arXiv:2512.20108
- bakker2020greedy — Greedy policy search MRI, NeurIPS 2020, arXiv:2010.16262
- bakker2022multicoil — Adaptive policies multi-coil, MIDL 2022, arXiv:2203.16392
- pineda2020rl — Active MR k-space sampling with RL, MICCAI 2020, arXiv:2007.10469
- zhang2019active — Reducing uncertainty w/ active acquisition, CVPR 2019, arXiv:1902.03051
- bahadir2019loupe — LOUPE, IPMI 2019 / IEEE TCI 2020
- huijben2020dps — Deep Probabilistic Subsampling, ICLR 2020
- vangorp2021adps — Active DPS, ICML 2021
- yin2021seqmri — End-to-end sequential sampling MRI, ML4H/PMLR 2021, arXiv:2105.06460
- vandecamp2023eig — Active subsampling w/ generative models + EIG, ICASSP 2023
- ravula2023masks — Diffusion-optimized (non-adaptive) CS-MRI masks, arXiv:2306.03284
- ayzenberg2026token — Token-uncertainty active MRI, arXiv:2603.21806
- goffinet2025diffnmr — DiffNMR2, arXiv:2502.05230
- vannierop2025cognitive — Patient-adaptive cognitive ultrasound, arXiv:2508.08782

## Diffusion inverse-problem solvers
- song2023pigdm — PiGDM, ICLR 2023 (OpenReview 9_gsMA8MRKQ; no arXiv)
- chung2023dps — DPS, ICLR 2023, arXiv:2209.14687
- kawar2022ddrm — DDRM, NeurIPS 2022, arXiv:2201.11793
- wang2023ddnm — DDNM (zero-shot restoration via null-space), ICLR 2023, arXiv:2212.00490
- jalal2021csgm — Robust CS MRI with generative priors, NeurIPS 2021
- chung2022scoremri — Score-based MRI, Chung & Ye, Medical Image Analysis 2022

## Posterior covariance in diffusion models (the estimator lineage — ADOPTED)
- peng2024covariance — Optimal posterior covariance, ICML 2024, arXiv:2402.02149
- boys2024tmpd — Tweedie Moment Projected Diffusions, TMLR 2024, arXiv:2310.06721
- hamidi2024cadps — CA-DPS, arXiv:2412.20045
- rissanen2025freehunch — Free Hunch, ICLR 2025, arXiv:2410.11149
- manor2024posterior — Posterior distribution in denoising, ICLR 2024, arXiv:2309.13598
- schioppa2026covaware — Covariance-aware sampling, arXiv:2605.13910

## Foundations
- efron2011tweedie — Tweedie's formula, Efron, JASA 2011
- hutchinson1990trace — Hutchinson stochastic trace estimator, Commun. Stat. 1990
- ho2020ddpm — DDPM, NeurIPS 2020
- song2021ddim — DDIM, ICLR 2021
- karras2022edm — EDM, NeurIPS 2022
- robbins1956empirical — Robbins empirical Bayes (optional, for Tweedie lineage)

## Data / domain
- zbontar2018fastmri — fastMRI, arXiv:1811.08839
- krizhevsky2009cifar — CIFAR-10 tech report
- karras2018progressive — CelebA-HQ (Progressive GAN), ICLR 2018
- sriram2020varnet — E2E-VarNet, MICCAI 2020
- uecker2014espirit — ESPIRiT, MRM 2014
- lustig2007sparse — Sparse MRI, MRM 2007
- wang2004ssim — SSIM, IEEE TIP 2004

## Added post-audit (verified 2026-07-07)
- christoffel2026dps — Christoffel-DPS: optimal sensor placement in DPS, arXiv:2605.06861
- divergence2026 — Divergence is Uncertainty (flow-matching posterior covariance), arXiv:2605.00941
