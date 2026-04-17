# FA-KGD: Frequency-Adaptive Kalman-Guided Diffusion for Accelerated MRI

**CS231N Final Project — Stanford, Spring 2025**

Accelerated MRI reconstruction using diffusion models with frequency-adaptive data consistency. We propose **FA-KGD**, which replaces the isotropic measurement correction in ΠGDM with a per-frequency Kalman gain calibrated from auto-calibration signal (ACS) lines, combined with a frequency-progressive data consistency (FPDC) schedule.

## Key idea

Standard diffusion-based MRI methods (DPS, ΠGDM) treat measurement noise as isotropic across k-space. In practice, k-space SNR drops with frequency — high-frequency components are noisier. FA-KGD exploits this structure:

1. **Frequency-adaptive Kalman gain** $K_i(t) = \sigma_t^2 / (\sigma_t^2 + \hat\sigma_i^2)$ — per-frequency noise estimates from ACS calibration
2. **FPDC schedule** — restricts data consistency to low frequencies at early (high-noise) steps, progressively expanding to full k-space
3. **Online M-step** (optional) — EMA update of $\hat\sigma_i^2$ from reconstruction residuals

## Results

| Method | R=4 PSNR (dB) | R=8 PSNR (dB) |
|--------|---------------|---------------|
| ΠGDM | 31.23 ± 0.73 | 30.85 ± 0.67 |
| **FA-KGD+FPDC** | **31.39 ± 0.74** | **31.00 ± 0.67** |
| Δ | **+0.17 ± 0.01** | **+0.15 ± 0.01** |

*5 slices, EDM score network (65M params, brain-trained), 20 diffusion steps, CPU inference.*

With oracle denoiser (isolating sampler quality): **+1.88 dB** at R=4.

## Project structure

```
fastmri/
├── src/
│   ├── samplers/           # Core sampling algorithms
│   │   ├── pigdm.py        # ΠGDM baseline (Song et al. 2023)
│   │   ├── fakgd.py        # FA-KGD + FPDC (ours)
│   │   ├── schedules.py    # DDPM and EDM noise schedules
│   │   └── mri_forward.py  # MRI forward model (FFT, masking)
│   ├── models/
│   │   ├── edm_loader.py   # EDM checkpoint loading + denoiser wrappers
│   │   └── unet.py         # Baseline U-Net
│   ├── data/               # Dataset and transforms
│   ├── training/           # Training loop
│   └── utils/
│       └── metrics.py      # NMSE, PSNR, SSIM
├── scripts/
│   └── reconstruct.py      # Main reconstruction script (oracle / EDM)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_reconstruction.ipynb
│   ├── 03_fakgd_verification.ipynb
│   ├── 04_closed_loop_em_simulation.ipynb
│   └── 05_edm_evaluation.ipynb
├── external/
│   └── adps/               # ADPS repo (ambient-diffusion-mri)
├── checkpoints/            # Pre-trained EDM models (gitignored)
├── configs/
│   └── default.yaml
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data

Download the [fastMRI](https://fastmri.med.nyu.edu/) singlecoil test set (~1.4 GB):

```bash
# Place HDF5 files in data/singlecoil_test/
ls data/singlecoil_test/*.h5 | head -3
```

### Pre-trained model

We use the supervised EDM checkpoint from [ADPS](https://github.com/utcsilab/ambient-diffusion-mri) (ICLR 2025):

```bash
# Clone ADPS (needed for model deserialization)
git clone --depth 1 https://github.com/utcsilab/ambient-diffusion-mri.git external/adps

# Download checkpoint (~1.7 GB zip → 250 MB pkl)
# Place in checkpoints/edm/supervised_R=1/network-snapshot.pkl
```

## Usage

### Reconstruction script

```bash
# Oracle mode (validates sampler logic, no GPU needed)
python scripts/reconstruct.py \
  --mode oracle --data_path data/singlecoil_test \
  --num_slices 10 --acceleration 4 --num_steps 100

# EDM mode (real score network)
python scripts/reconstruct.py \
  --mode edm --checkpoint_dir checkpoints/edm/supervised_R=1 \
  --data_path data/singlecoil_test \
  --num_slices 5 --acceleration 4 --num_steps 20 \
  --schedule edm --sigma_max 10 \
  --target_resolution 384 320
```

### Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | Data exploration | HDF5 structure, k-space visualization, undersampling artifacts |
| 02 | Baseline reconstruction | Zero-filled baseline, U-Net sanity check, PSNR/SSIM metrics |
| 03 | FA-KGD verification | Frequency-dependent noise profiles, $AA^\top$ diagonality, adaptive gain maps |
| 04 | Closed-loop EM simulation | Oracle denoiser: ΠGDM vs FA-KGD vs FA-KGD-EM, $\hat\sigma^2$ convergence |
| 05 | EDM evaluation | Real model results, ablations, visualization of reconstructions |

## Method details

The reverse diffusion update at step $t$ replaces the isotropic ΠGDM correction:

$$x_{t-1} = \mu_\theta(x_t) + \mathcal{F}^{-1}\left[K(t) \odot M \odot (y - \mathcal{F}[\mu_\theta(x_t)])\right] + \sigma_{t-1}\epsilon$$

where the Kalman gain is:

$$K_i(t) = \frac{\sigma_t^2}{\sigma_t^2 + \hat\sigma_i^2}$$

FPDC progressively expands the data consistency radius:

$$r(t) = r_{\text{ACS}} + (r_{\max} - r_{\text{ACS}}) \left(\frac{t}{T}\right)^\beta$$

## References

- Zbontar et al. "fastMRI: An Open Dataset and Benchmarks for Accelerated MRI" (2018)
- Song et al. "Pseudoinverse-Guided Diffusion Models for Inverse Problems" (ICLR 2023)
- Levac et al. "Ambient Diffusion Posterior Sampling" (ICLR 2025)
- Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022)
