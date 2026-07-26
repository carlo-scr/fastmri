"""Multi-coil reconstruction with legacy PVAS and active-line baselines.

Sister script to ``scripts/reconstruct.py``. Operates on raw multi-coil
k-space from fastMRI brain (and similar), and supports the full
PV/active-sampling family. The ``pv`` and ``pv_active`` methods retain the
pre-audit single-Fourier estimator for backward compatibility; they are not
valid operator-space SENSE covariance baselines. Use the SketchSense methods
for new paper runs once integrated.

Methods (``--methods`` list, any subset):
  pigdm_mc     : multi-coil ΠGDM with per-coil scalar noise
  fakgd_mc     : multi-coil FA-KGD+FPDC (per-frequency Kalman gain)
  pv           : (A) PV-gated FA-KGD only
  pv_active    : (A) + (D) active line acquisition
  operator_diag_active
                : corrected full-SENSE cross-Hutchinson marginal selection
  sketch_marginal_active
                : PSD operator sketch scored one line at a time
  sketch_logdet_active
                : PSD operator sketch with correlation-aware joint batches
  static_match : ΠGDM-MC at the matched-budget static mask (a fair
                 baseline for the active runs; budget = R-mask + active_lines)

Example:
  python scripts/reconstruct_mc.py \
    --checkpoint_dir checkpoints/edm/supervised_R=1 \
    --data_path data/multicoil_val \
    --num_volumes 5 --num_slices 8 \
    --acceleration 4 --center_fraction 0.08 \
    --num_steps 20 --schedule edm --sigma_max 80 \
    --target_resolution 384 320 \
    --methods pigdm_mc fakgd_mc pv pv_active static_match \
    --active_lines 16 --active_rounds 2 \
    --device cuda --output_dir outputs/mc_eval
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_fn

# Repo on path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.samplers.mri_forward import fft2c, ifft2c, create_mask, build_radius_grid
from src.samplers.schedules import edm_sigma_schedule, ddpm_sigma_schedule
from src.samplers.sense import (
    MultiCoilSENSE, sense_combine,
    estimate_sens_maps_lowres, estimate_noise_per_coil,
)
from src.samplers.multicoil import run_pigdm_mc, run_fakgd_mc
from src.samplers.multicoil_pv import (
    random_complement_order,
    run_ensemble_active_mc,
    run_fakgd_mc_pv,
)
from src.models.edm_loader import load_edm_model, EDMDenoiser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Center-crop (and zero-pad if input is smaller) so the result is exactly h×w."""
    H, W = x.shape[-2], x.shape[-1]
    # Pad first if smaller than target (some fastMRI volumes are <384x320)
    pad_h = max(0, h - H)
    pad_w = max(0, w - W)
    if pad_h or pad_w:
        ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
        pw0, pw1 = pad_w // 2, pad_w - pad_w // 2
        # F.pad takes (W_left, W_right, H_top, H_bottom) for last 2 dims
        from torch.nn.functional import pad as _pad
        x = _pad(x, (pw0, pw1, ph0, ph1))
        H, W = x.shape[-2], x.shape[-1]
    sh = (H - h) // 2
    sw = (W - w) // 2
    return x[..., sh:sh + h, sw:sw + w]


def _psnr(gt_mag: torch.Tensor, rec_mag: torch.Tensor, eval_size: int = 320) -> float:
    H = min(gt_mag.shape[-2], eval_size)
    W = min(gt_mag.shape[-1], eval_size)
    g = _center_crop(gt_mag, H, W)
    r = _center_crop(rec_mag, H, W)
    dr = g.max()
    mse = ((g - r) ** 2).mean()
    if mse < 1e-12:
        return 100.0
    return float(10 * torch.log10(dr ** 2 / mse))


def _ssim(gt_mag: torch.Tensor, rec_mag: torch.Tensor, eval_size: int = 320) -> float:
    H = min(gt_mag.shape[-2], eval_size)
    W = min(gt_mag.shape[-1], eval_size)
    g = _center_crop(gt_mag, H, W).cpu().numpy()
    r = _center_crop(rec_mag, H, W).cpu().numpy()
    dr = float(g.max() - g.min())
    if dr < 1e-12:
        return 0.0
    return float(ssim_fn(g, r, data_range=dr))


def _nmse(gt_mag: torch.Tensor, rec_mag: torch.Tensor, eval_size: int = 320) -> float:
    """Normalized squared error on the same center crop as PSNR/SSIM."""
    H = min(gt_mag.shape[-2], eval_size)
    W = min(gt_mag.shape[-1], eval_size)
    g = _center_crop(gt_mag, H, W)
    r = _center_crop(rec_mag, H, W)
    numerator = (g - r).square().sum()
    denominator = g.square().sum().clamp_min(1e-12)
    return float(numerator / denominator)


def _static_matched_mask(
    base_mask: torch.Tensor,
    extra_cols: int,
    seed: int,
) -> torch.Tensor:
    """Extend ``base_mask`` by the random policy's exact final complement.

    This creates a timing-only control: the static and mid-trajectory random
    baselines start from the same mask, end with the same mask, and differ only
    in when the extra measurements become available.
    """
    out = base_mask.clone()
    order = random_complement_order(base_mask, seed)
    chosen = order[: max(0, int(extra_cols))]
    if chosen:
        out[:, chosen] = 1.0
    return out


def _add_kspace_noise(mc_k: torch.Tensor, sig_c: torch.Tensor,
                      seed: int) -> torch.Tensor:
    """Add white per-coil Gaussian noise (variance sig_c[c]) to mc_k."""
    Nc = mc_k.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    nz = (torch.randn(mc_k.shape, generator=g)
          + 1j * torch.randn(mc_k.shape, generator=g)) / np.sqrt(2)
    nz = nz.to(mc_k.device)
    return mc_k + (sig_c.to(mc_k.device).sqrt().view(Nc, 1, 1) * nz).to(mc_k.dtype)


def _load_volume(h5_path: str, target_shape: tuple[int, int]):
    """Load multi-coil k-space + RSS GT for every slice in a volume.

    Returns:
        list of (mc_k_clean, rss_gt) tuples; mc_k_clean is (Nc,H,W) complex
        with H,W = target_shape; rss_gt is (H,W) real.
    """
    out = []
    with h5py.File(h5_path, 'r') as f:
        ks_all = f['kspace'][:]                  # (Nz,Nc,H0,W0)
        if 'reconstruction_rss' in f:
            rss_all = f['reconstruction_rss'][:]
        else:
            rss_all = None
    Nz = ks_all.shape[0]
    H, W = target_shape
    for sl in range(Nz):
        ks = torch.from_numpy(ks_all[sl])         # (Nc,H0,W0) complex
        coil_imgs = ifft2c(ks)
        coil_imgs = _center_crop(coil_imgs, H, W)
        mc_k = fft2c(coil_imgs)
        if rss_all is not None and rss_all.shape[-1] >= W and rss_all.shape[-2] >= H:
            rss = torch.from_numpy(rss_all[sl])
            rss = _center_crop(rss, H, W)
        else:
            rss = torch.sqrt((coil_imgs.abs() ** 2).sum(0))
        out.append((mc_k.to(torch.complex64), rss.float()))
    return out


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def subprocess_git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=project_root,
            text=True,
        ).strip()
    except Exception:
        return 'unknown'


def _paired_bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    seed: int,
    resamples: int = 10_000,
) -> tuple[float, float]:
    """Deterministic percentile interval for independent volume effects."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('differences must be a nonempty one-dimensional array')
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0, values.size, size=(int(resamples), values.size)
    )
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Per-slice runner
# ---------------------------------------------------------------------------

def run_slice(args, denoiser, sigma_schedule, mc_k_clean, rss_gt,
              file_seed: int, slice_idx: int, device: torch.device) -> dict:
    """Reconstruct one slice with all requested methods. Returns metrics dict."""
    Nc, H, W = mc_k_clean.shape
    # Vary stochastic state across slices while sharing it across methods on
    # the same slice. This avoids pseudoreplication from identical masks,
    # latents, re-noising paths, and probe streams on adjacent slices.
    slice_seed = int(
        (int(args.seed) * 1_000_003 + int(file_seed) + int(slice_idx) * 7_919)
        % (2**63 - 1)
    )
    slice_mask_seed = int(
        (int(args.mask_seed) * 65_537 + int(file_seed) + int(slice_idx) * 104_729)
        % (2**32 - 1)
    )

    # Normalise so that |RSS| has max 1.0 (consistent with EDM training)
    scale = float(rss_gt.max()).__pow__(1.0)
    if scale < 1e-12:
        scale = 1.0
    mc_k = mc_k_clean / scale
    rss = (rss_gt / scale).to(device)

    # Prospective acquisition cannot estimate noise from unacquired full-scan
    # corners. Production runs therefore use a known synthetic complex-noise
    # standard deviation. The former full-scan estimate is available only as
    # an explicitly labeled legacy ablation.
    if args.legacy_full_scan_noise:
        sig_c = estimate_noise_per_coil(mc_k).to(device)
        noise_protocol = "legacy_full_scan_corner_estimate"
    else:
        known_var = max(float(args.noise_scale) ** 2, 1e-12)
        sig_c = torch.full((Nc,), known_var, device=device)
        noise_protocol = "known_synthetic"
    # Add measurement noise (deterministic per slice)
    noise_seed = args.noise_seed + file_seed + slice_idx * 7919
    mc_kn = _add_kspace_noise(mc_k, sig_c, seed=noise_seed).to(device)
    mc_k = mc_k.to(device)

    # R-mask (1D, expanded to 2D)
    mask_R = create_mask(W, args.center_fraction, args.acceleration,
                         seed=slice_mask_seed).expand(H, -1).to(device)
    n_R_cols = int(mask_R[0].sum().item())

    # Sens maps from low-res ACS of the (noisy) acquired data
    sens = estimate_sens_maps_lowres(mc_kn, center_fraction=args.center_fraction).to(device)
    sense_op = MultiCoilSENSE(mask_R, sens, sig_c).to(device)

    # SENSE-oracle reference (best-possible SENSE combine of the clean coil images)
    sense_oracle = sense_combine(ifft2c(mc_k), sens).abs()

    # σ_i² init: flat = mean(σ_c²)  (white in k-space → matches reality)
    sigma_i_sq = torch.full((H, W), float(sig_c.mean()), device=device)

    # FPDC parameters (used by fakgd_mc)
    radius_grid = build_radius_grid(H, W).to(device)
    r_max = radius_grid.max().item()
    r_acs = args.center_fraction * r_max

    out: dict = {
        'status': 'CORRECTED_RUN_UNREVIEWED',
        'n_R_cols': n_R_cols,
        'sampler_seed': slice_seed,
        'mask_seed': slice_mask_seed,
        'noise_seed': int(noise_seed),
        'noise_protocol': noise_protocol,
        'complex_noise_variance_per_coil': sig_c.detach().cpu().tolist(),
        'methods': {},
    }

    def _reset_peak_memory() -> None:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def _peak_memory_mb() -> float | None:
        if device.type != 'cuda':
            return None
        torch.cuda.synchronize(device)
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

    def _elapsed(started_at: float) -> float:
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        return float(time.time() - started_at)

    canonical_cache: dict[tuple[int, ...], dict] = {}

    def _canonical_replay(final_mask: torch.Tensor) -> dict | None:
        """Evaluate a selected mask with one common passive PiGDM solver."""
        if args.skip_canonical_replay:
            return None
        selected_columns = tuple(
            torch.nonzero(
                final_mask.sum(dim=0) > 0, as_tuple=False
            ).flatten().cpu().tolist()
        )
        if selected_columns in canonical_cache:
            return canonical_cache[selected_columns]
        replay_operator = MultiCoilSENSE(final_mask, sens, sig_c).to(device)
        replay_start = time.time()
        replay_nfe_start = getattr(denoiser, 'nfe', None)
        replay = run_pigdm_mc(
            y_mc=final_mask * mc_kn,
            sense_op=replay_operator,
            sigma_schedule=sigma_schedule,
            denoiser_fn=denoiser,
            sigma_y=None,
            seed=slice_seed,
        )
        replay_nfe = (
            int(denoiser.nfe) - int(replay_nfe_start)
            if replay_nfe_start is not None and hasattr(denoiser, 'nfe')
            else None
        )
        replay_mag = replay['recon'].abs()
        record = {
            'canonical_psnr_rss': _psnr(
                rss, replay_mag, eval_size=args.eval_size
            ),
            'canonical_psnr_sense': _psnr(
                sense_oracle, replay_mag, eval_size=args.eval_size
            ),
            'canonical_ssim_rss': _ssim(
                rss, replay_mag, eval_size=args.eval_size
            ),
            'canonical_ssim_sense': _ssim(
                sense_oracle, replay_mag, eval_size=args.eval_size
            ),
            'canonical_nmse_rss': _nmse(
                rss, replay_mag, eval_size=args.eval_size
            ),
            'canonical_nmse_sense': _nmse(
                sense_oracle, replay_mag, eval_size=args.eval_size
            ),
            'canonical_runtime_s': float(time.time() - replay_start),
            'canonical_nfe': replay_nfe,
        }
        canonical_cache[selected_columns] = record
        return record

    def _record(
        name,
        recon,
        total_cols,
        runtime,
        diagnostics=None,
        nfe_start=None,
        final_mask=None,
    ):
        rec_mag = recon.abs() if torch.is_complex(recon) else recon
        rec_mag = rec_mag.to(device)
        rec = {
            'psnr_rss':    _psnr(rss, rec_mag, eval_size=args.eval_size),
            'psnr_sense':  _psnr(sense_oracle, rec_mag, eval_size=args.eval_size),
            'ssim_rss':    _ssim(rss, rec_mag, eval_size=args.eval_size),
            'ssim_sense':  _ssim(sense_oracle, rec_mag, eval_size=args.eval_size),
            'nmse_rss':    _nmse(rss, rec_mag, eval_size=args.eval_size),
            'nmse_sense':  _nmse(sense_oracle, rec_mag, eval_size=args.eval_size),
            'budget_cols': int(total_cols),
            'runtime_s':   float(runtime),
            'peak_cuda_memory_mb': _peak_memory_mb(),
        }
        if nfe_start is not None and hasattr(denoiser, 'nfe'):
            rec['nfe'] = int(denoiser.nfe) - int(nfe_start)
        if final_mask is not None:
            selected_columns = torch.nonzero(
                final_mask.sum(dim=0) > 0, as_tuple=False
            ).flatten()
            rec['final_columns'] = selected_columns.cpu().tolist()
            if int(selected_columns.numel()) != int(total_cols):
                raise AssertionError(
                    f"{name} reports {total_cols} columns but its final mask "
                    f"contains {selected_columns.numel()}"
                )
            canonical = _canonical_replay(final_mask)
            if canonical is not None:
                rec.update(canonical)
        if diagnostics is not None:
            kdiv = diagnostics.get('K_div_vs_pigdm') or []
            pang = diagnostics.get('P_ang_frac') or []
            if kdiv:
                rec['K_div_mean'] = float(np.mean(kdiv))
                rec['K_div_max']  = float(np.max(kdiv))
            if pang:
                rec['P_ang_frac_mean'] = float(np.mean(pang))
                rec['P_ang_frac_max']  = float(np.max(pang))
            sketch_rows = diagnostics.get('operator_sketches') or []
            if sketch_rows:
                rec['sketch_effective_rank_mean'] = float(np.mean([
                    row['effective_rank'] for row in sketch_rows
                ]))
                rec['sketch_core_asymmetry_mean'] = float(np.mean([
                    row['core_asymmetry'] for row in sketch_rows
                ]))
                rec['sketch_negative_mass_mean'] = float(np.mean([
                    row['core_negative_mass'] for row in sketch_rows
                ]))
        out['methods'][name] = rec

    def _actual_cols(result, fallback):
        mask_final = result.get('mask_final')
        if mask_final is None:
            return int(fallback)
        return int((mask_final.sum(dim=0) > 0).sum().item())

    y_R = mask_R * mc_kn

    # ---- pigdm_mc ----
    if 'pigdm_mc' in args.methods:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_pigdm_mc(y_mc=y_R, sense_op=sense_op,
                         sigma_schedule=sigma_schedule,
                         denoiser_fn=denoiser, sigma_y=None, seed=slice_seed)
        _record('pigdm_mc', r['recon'], n_R_cols, _elapsed(t0),
                nfe_start=nfe0, final_mask=mask_R)

    # ---- fakgd_mc (FPDC, no PV) ----
    if 'fakgd_mc' in args.methods:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc(y_mc=y_R, sense_op=sense_op,
                         sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                         sigma_i_sq_init=sigma_i_sq, r_acs=r_acs, r_max=r_max,
                         beta_fpdc=args.beta_fpdc, alpha_ema=args.alpha_ema,
                         gamma=args.gamma, m_step_mode=args.m_step_mode,
                         seed=slice_seed)
        _record('fakgd_mc', r['recon'], n_R_cols, _elapsed(t0),
                nfe_start=nfe0, final_mask=mask_R)

    # ---- pv (PV-gated only) ----
    if 'pv' in args.methods:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=True, refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=0, seed=slice_seed,
                            return_diagnostics=True)
        _record('pv', r['recon'], n_R_cols, _elapsed(t0),
                diagnostics=r.get('diagnostics'), nfe_start=nfe0,
                final_mask=mask_R)

    # ---- pv_active (PV + active acquisition) ----
    if 'pv_active' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=(args.active_gate == 'on'), refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='pv',
                            y_mc_full=mc_kn, seed=slice_seed,
                            return_diagnostics=True)
        _record('pv_active', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), diagnostics=r.get('diagnostics'),
                nfe_start=nfe0, final_mask=r.get('mask_final'))

    # ---- corrected full-operator and SketchSense acquisition policies ----
    sketch_methods = {
        'operator_diag_active': 'operator_diag',
        'sketch_marginal_active': 'sketch_marginal',
        'sketch_logdet_active': 'sketch_logdet',
    }
    for method_name, active_score in sketch_methods.items():
        if method_name not in args.methods or args.active_lines <= 0:
            continue
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(
            y_mc=y_R,
            sense_op=sense_op,
            sigma_schedule=sigma_schedule,
            denoiser_fn=denoiser,
            sigma_i_sq_init=sigma_i_sq,
            # Acquisition-only comparison: legacy single-Fourier PV gating is
            # deliberately disabled for every operator-sketch policy.
            use_pv_gate=False,
            refine_sens=False,
            pv_centered=True,
            active_lines=args.active_lines,
            active_rounds=args.active_rounds,
            active_after_frac=args.active_after_frac,
            active_until_frac=args.active_until_frac,
            active_score=active_score,
            sketch_rank=args.sketch_rank,
            sketch_fd_step=args.sketch_fd_step,
            sketch_psd_rtol=args.sketch_psd_rtol,
            y_mc_full=mc_kn,
            seed=slice_seed,
            return_diagnostics=True,
        )
        _record(
            method_name,
            r['recon'],
            _actual_cols(r, n_R_cols + args.active_lines),
            _elapsed(t0),
            diagnostics=r.get('diagnostics'),
            nfe_start=nfe0,
            final_mask=r.get('mask_final'),
        )

    # ---- random_adaptive (mid-diffusion uniform-random acquisition) ----
    # Same multi-round schedule and budget as PVAS; columns picked uniformly
    # at random instead of by PV score. Isolates the value of variance-guided
    # selection (vs. "any mid-diffusion adaptation").
    if 'random_adaptive' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=(args.active_gate == 'on'), refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='random',
                            y_mc_full=mc_kn, seed=slice_seed)
        _record('random_adaptive', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- energy_active (image-conditional, P-free; primary new score) ----
    # Score(c) = sum_{c,r} |F(S * mu_theta(x_t))[c,r,col]|^2 over unsampled
    # cols. Uses the diffusion prior's predicted k-space energy directly,
    # bypassing the (saturated) PV gate. No oracle access; legal in real MRI.
    if 'energy_active' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=(args.active_gate == 'on'), refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='energy',
                            y_mc_full=mc_kn, seed=slice_seed)
        _record('energy_active', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- oracle_active (retrospective greedy reference) ----
    # Score(c) = sum_{c,r} |y_full - F(S*mu)|^2 over unsampled cols.
    # Requires y_mc_full and is not deployable. It is not an upper bound on
    # final reconstruction utility.
    if 'oracle_active' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=(args.active_gate == 'on'), refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='residual_oracle',
                            y_mc_full=mc_kn, seed=slice_seed)
        _record('oracle_active', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- equi_adaptive (mid-diffusion gap-filling acquisition) ----
    # Deterministic non-PV baseline: greedily fill the largest column gaps.
    if 'equi_adaptive' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=(args.active_gate == 'on'), refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            pv_radial_smooth=args.pv_radial_smooth,
                            pv_radial_degree=args.pv_radial_degree,
                            pv_normalize=args.pv_normalize,
                            pv_n_probes=args.pv_n_probes,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='equispaced',
                            y_mc_full=mc_kn, seed=slice_seed)
        _record('equi_adaptive', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- ens_active (shared-pipeline particle-covariance control) ----
    # N parallel PiGDM particles (scalar gain); columns scored by cross-
    # particle variance of predicted k-space; recon = particle mean.
    # Faithful-cost ensemble reference: NFE = N*T per slice.
    if 'ens_active' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_ensemble_active_mc(y_mc=y_R, sense_op=sense_op,
                                   sigma_schedule=sigma_schedule,
                                   denoiser_fn=denoiser,
                                   sigma_i_sq_init=sigma_i_sq,
                                   n_particles=args.ens_particles,
                                   active_lines=args.active_lines,
                                   active_rounds=args.active_rounds,
                                   active_after_frac=args.active_after_frac,
                                   active_until_frac=args.active_until_frac,
                                   active_score='variance',
                                   y_mc_full=mc_kn, seed=slice_seed)
        _record('ens_active', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- ens_random_active (particle mean + random selection) ----
    # Isolates the particle-averaging benefit from ensemble selection.
    if 'ens_random_active' in args.methods and args.active_lines > 0:
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_ensemble_active_mc(y_mc=y_R, sense_op=sense_op,
                                   sigma_schedule=sigma_schedule,
                                   denoiser_fn=denoiser,
                                   sigma_i_sq_init=sigma_i_sq,
                                   n_particles=args.ens_particles,
                                   active_lines=args.active_lines,
                                   active_rounds=args.active_rounds,
                                   active_after_frac=args.active_after_frac,
                                   active_until_frac=args.active_until_frac,
                                   active_score='random',
                                   y_mc_full=mc_kn, seed=slice_seed)
        _record('ens_random_active', r['recon'],
                _actual_cols(r, n_R_cols + args.active_lines),
                _elapsed(t0), nfe_start=nfe0,
                final_mask=r.get('mask_final'))

    # ---- static_match (matched-budget static-random baseline) ----
    if 'static_match' in args.methods and args.active_lines > 0:
        mask_static = _static_matched_mask(
            mask_R, args.active_lines, seed=slice_seed,
        )
        budget = int((mask_static.sum(dim=0) > 0).sum().item())
        op_static = MultiCoilSENSE(mask_static, sens, sig_c).to(device)
        _reset_peak_memory()
        t0 = time.time()
        nfe0 = getattr(denoiser, 'nfe', None)
        r = run_pigdm_mc(y_mc=mask_static * mc_kn, sense_op=op_static,
                         sigma_schedule=sigma_schedule,
                         denoiser_fn=denoiser, sigma_y=None, seed=slice_seed)
        _record('static_match', r['recon'], budget, _elapsed(t0),
                nfe_start=nfe0, final_mask=mask_static)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument(
        '--dataset_revision',
        type=str,
        default='unspecified',
        help='Dataset release/snapshot identifier recorded in results. '
             'Production artifacts must not leave this as unspecified.',
    )
    p.add_argument(
        '--skip_file_hashes',
        action='store_true',
        help='Pilot-only speed option. Production runs hash every input volume.',
    )
    p.add_argument('--num_volumes', type=int, default=2)
    p.add_argument('--num_slices', type=int, default=0,
                   help='Slices per volume (0 = all)')
    p.add_argument('--target_resolution', type=int, nargs=2, default=[384, 320])
    p.add_argument('--eval_size', type=int, default=320,
                   help='Center-crop size for PSNR/SSIM evaluation.')

    # Acquisition / mask
    p.add_argument('--acceleration', type=int, default=4)
    p.add_argument('--center_fraction', type=float, default=0.08)
    p.add_argument('--mask_seed', type=int, default=42)

    # Noise
    p.add_argument('--noise_scale', type=float, default=0.01,
                   help='Known synthetic complex-noise std used for controlled '
                        'prospective experiments (default: 0.01 after per-slice '
                        'normalization).')
    p.add_argument('--legacy_full_scan_noise', action='store_true',
                   help='Legacy-only: estimate per-coil variance from complete '
                        'unacquired k-space corners. Invalid for prospective '
                        'active-sensing claims.')
    p.add_argument('--noise_seed', type=int, default=12345)

    # Diffusion schedule
    p.add_argument('--schedule', type=str, default='edm', choices=['edm', 'ddpm'])
    p.add_argument('--num_steps', type=int, default=20)
    p.add_argument('--sigma_min', type=float, default=0.002)
    p.add_argument('--sigma_max', type=float, default=80.0)

    # Model
    p.add_argument('--checkpoint_dir', type=str, required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)

    # FA-KGD (FPDC) params
    p.add_argument('--beta_fpdc', type=float, default=1.0)
    p.add_argument('--alpha_ema', type=float, default=0.95)
    p.add_argument('--gamma', type=float, default=0.0)
    p.add_argument('--m_step_mode', type=str, default='clamp',
                   choices=['full', 'clamp', 'off'])

    # PV gate params
    p.add_argument('--pv_eps_probe', type=float, default=5e-2)
    p.add_argument('--pv_centered', action='store_true', default=True)
    p.add_argument('--no_pv_centered', dest='pv_centered', action='store_false')
    # Radial smoothing destroys the angular structure of P that PV is
    # supposed to expose; default OFF after the v1 post-mortem (PV ≡ PiGDM
    # under smoothing+normalize). Re-enable with --pv_radial_smooth to
    # reproduce v1.
    p.add_argument('--pv_radial_smooth', dest='pv_radial_smooth',
                   action='store_true', default=False)
    p.add_argument('--no_pv_radial_smooth', dest='pv_radial_smooth',
                   action='store_false')
    p.add_argument('--pv_radial_degree', type=int, default=4)
    # Normalization rescales P so mean(P)=2*sigma_t^2 — helpful when P is noisy,
    # harmful when P actually has structure (it cancels with sigma_ci^2 in K).
    p.add_argument('--pv_normalize', dest='pv_normalize',
                   action='store_true', default=False)
    p.add_argument('--no_pv_normalize', dest='pv_normalize',
                   action='store_false')
    # Multi-probe Hutchinson averaging — reduces P variance so the angular
    # structure isn't drowned out by single-probe noise.
    p.add_argument('--pv_n_probes', type=int, default=2)

    # Active sampling params
    p.add_argument('--active_lines', type=int, default=0)
    p.add_argument('--active_rounds', type=int, default=1)
    p.add_argument('--active_after_frac', type=float, default=0.5)
    p.add_argument('--active_until_frac', type=float, default=0.85)
    p.add_argument('--active_gate', choices=['on', 'off'], default='off',
                   help="'off' = selection-only active methods: scalar PiGDM "
                        "gain everywhere, PV computed only at acquisition "
                        "rounds (NFE = T + 2*pv_n_probes*active_rounds).")
    p.add_argument('--ens_particles', type=int, default=4,
                   help='Particles for the ens_active / ens_random_active '
                        'ADS-style ensemble baselines (NFE = N*T).')
    p.add_argument('--sketch_rank', type=int, default=4,
                   help='Number of real-embedded operator probes for '
                        'operator_diag/sketch policies.')
    p.add_argument('--sketch_fd_step', type=float, default=5e-2,
                   help='Central finite-difference step as a fraction of the '
                        'current diffusion sigma after pullback normalization.')
    p.add_argument('--sketch_psd_rtol', type=float, default=1e-6,
                   help='Relative eigenvalue floor for the Nyström core.')
    p.add_argument(
        '--skip_canonical_replay',
        action='store_true',
        help='Pilot-only speed option. By default each final mask is replayed '
             'with the same passive single-trajectory PiGDM solver so mask '
             'quality is separated from the online reconstruction method.',
    )

    # Methods
    p.add_argument('--methods', type=str, nargs='+',
                   default=['pigdm_mc', 'fakgd_mc', 'pv', 'pv_active',
                            'random_adaptive', 'static_match'],
                   choices=['pigdm_mc', 'fakgd_mc', 'pv', 'pv_active',
                            'operator_diag_active',
                            'sketch_marginal_active',
                            'sketch_logdet_active',
                            'energy_active', 'oracle_active',
                            'random_adaptive', 'equi_adaptive', 'static_match',
                            'ens_active', 'ens_random_active'])

    # Output
    p.add_argument('--output_dir', type=str, required=True)

    args = p.parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    if args.device == 'mps' and not torch.backends.mps.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Schedule
    if args.schedule == 'edm':
        sigma_schedule = edm_sigma_schedule(args.num_steps,
                                            sigma_min=args.sigma_min,
                                            sigma_max=args.sigma_max).to(device)
    else:
        sigma_schedule = ddpm_sigma_schedule(args.num_steps).to(device)

    # Model
    print('Loading EDM model...', flush=True)
    net = load_edm_model(args.checkpoint_dir, method='auto', device=str(device))
    denoiser = EDMDenoiser(net, device=str(device))

    # Files
    files = sorted(glob.glob(os.path.join(args.data_path, '*.h5')))[:args.num_volumes]
    if not files:
        raise FileNotFoundError(f'No .h5 files in {args.data_path}')
    print(f'Found {len(files)} volumes', flush=True)

    target = tuple(args.target_resolution)
    all_results = []
    volume_manifest = []

    for fi, fp in enumerate(files):
        fname = Path(fp).name
        file_sha256 = (
            None if args.skip_file_hashes else _sha256_file(fp)
        )
        volume_manifest.append({
            'volume': fname,
            'sha256': file_sha256,
        })
        file_seed = int.from_bytes(
            hashlib.blake2b(fname.encode(), digest_size=4).digest(),
            'little')
        try:
            slices = _load_volume(fp, target)
        except Exception as e:
            print(f'  SKIP {fname}: {e}', flush=True)
            continue

        if args.num_slices > 0:
            mid = len(slices) // 2
            half = args.num_slices // 2
            sl_idx = list(range(max(0, mid - half),
                                min(len(slices), mid - half + args.num_slices)))
        else:
            sl_idx = list(range(len(slices)))

        for si in sl_idx:
            mc_k_clean, rss_gt = slices[si]
            t_slice = time.time()
            res = run_slice(args, denoiser, sigma_schedule,
                            mc_k_clean, rss_gt, file_seed, si, device)
            res.update({
                'volume': fname,
                'volume_sha256': file_sha256,
                'slice': si,
            })
            all_results.append(res)
            # Pretty per-slice line
            line = f'[{fi+1}/{len(files)}] {fname} sl={si} ({time.time()-t_slice:.0f}s):'
            for m, v in res['methods'].items():
                line += f'  {m}={v["psnr_sense"]:.2f}'
            print(line, flush=True)

    # Aggregate
    print('\n' + '=' * 70)
    print(f'AGGREGATE  ({len(all_results)} slices)')
    print('=' * 70)
    method_names = []
    for r in all_results:
        for m in r['methods']:
            if m not in method_names:
                method_names.append(m)

    summary = {}
    for m in method_names:
        psnr_rss   = np.array([r['methods'][m]['psnr_rss']   for r in all_results if m in r['methods']])
        psnr_sense = np.array([r['methods'][m]['psnr_sense'] for r in all_results if m in r['methods']])
        ssim_rss   = np.array([r['methods'][m]['ssim_rss']   for r in all_results if m in r['methods']])
        ssim_sense = np.array([r['methods'][m]['ssim_sense'] for r in all_results if m in r['methods']])
        selection_psnr_rss = np.array([
            r['methods'][m].get(
                'canonical_psnr_rss', r['methods'][m]['psnr_rss']
            )
            for r in all_results if m in r['methods']
        ])
        selection_psnr_sense = np.array([
            r['methods'][m].get(
                'canonical_psnr_sense', r['methods'][m]['psnr_sense']
            )
            for r in all_results if m in r['methods']
        ])
        selection_ssim_rss = np.array([
            r['methods'][m].get(
                'canonical_ssim_rss', r['methods'][m]['ssim_rss']
            )
            for r in all_results if m in r['methods']
        ])
        selection_ssim_sense = np.array([
            r['methods'][m].get(
                'canonical_ssim_sense', r['methods'][m]['ssim_sense']
            )
            for r in all_results if m in r['methods']
        ])
        nmse_rss = np.array([
            r['methods'][m]['nmse_rss']
            for r in all_results if m in r['methods']
        ])
        nmse_sense = np.array([
            r['methods'][m]['nmse_sense']
            for r in all_results if m in r['methods']
        ])
        selection_nmse_rss = np.array([
            r['methods'][m].get(
                'canonical_nmse_rss', r['methods'][m]['nmse_rss']
            )
            for r in all_results if m in r['methods']
        ])
        selection_nmse_sense = np.array([
            r['methods'][m].get(
                'canonical_nmse_sense', r['methods'][m]['nmse_sense']
            )
            for r in all_results if m in r['methods']
        ])
        budget     = np.array([r['methods'][m]['budget_cols'] for r in all_results if m in r['methods']])
        nfe         = np.array([r['methods'][m].get('nfe', np.nan)
                                for r in all_results if m in r['methods']])
        runtime = np.array([
            r['methods'][m]['runtime_s']
            for r in all_results if m in r['methods']
        ])
        peak_memory = np.array([
            r['methods'][m].get('peak_cuda_memory_mb', np.nan)
            if r['methods'][m].get('peak_cuda_memory_mb') is not None
            else np.nan
            for r in all_results if m in r['methods']
        ])
        summary[m] = {
            'psnr_rss_mean':   float(psnr_rss.mean()),   'psnr_rss_std':   float(psnr_rss.std()),
            'psnr_sense_mean': float(psnr_sense.mean()), 'psnr_sense_std': float(psnr_sense.std()),
            'ssim_rss_mean':   float(ssim_rss.mean()),   'ssim_rss_std':   float(ssim_rss.std()),
            'ssim_sense_mean': float(ssim_sense.mean()), 'ssim_sense_std': float(ssim_sense.std()),
            'nmse_rss_mean':   float(nmse_rss.mean()),   'nmse_rss_std':   float(nmse_rss.std()),
            'nmse_sense_mean': float(nmse_sense.mean()), 'nmse_sense_std': float(nmse_sense.std()),
            'selection_psnr_rss_mean': float(selection_psnr_rss.mean()),
            'selection_psnr_rss_std': float(selection_psnr_rss.std()),
            'selection_psnr_sense_mean': float(selection_psnr_sense.mean()),
            'selection_psnr_sense_std': float(selection_psnr_sense.std()),
            'selection_ssim_rss_mean': float(selection_ssim_rss.mean()),
            'selection_ssim_rss_std': float(selection_ssim_rss.std()),
            'selection_ssim_sense_mean': float(selection_ssim_sense.mean()),
            'selection_ssim_sense_std': float(selection_ssim_sense.std()),
            'selection_nmse_rss_mean': float(selection_nmse_rss.mean()),
            'selection_nmse_rss_std': float(selection_nmse_rss.std()),
            'selection_nmse_sense_mean': float(selection_nmse_sense.mean()),
            'selection_nmse_sense_std': float(selection_nmse_sense.std()),
            'selection_metric_source': (
                'system reconstruction'
                if args.skip_canonical_replay
                else 'canonical passive replay'
            ),
            'budget_cols':     int(budget.mean()),
            'nfe_mean':        float(np.nanmean(nfe)),
            'runtime_s_mean':  float(runtime.mean()),
            'runtime_s_std':   float(runtime.std()),
            'peak_cuda_memory_mb_mean': (
                float(np.nanmean(peak_memory))
                if bool(np.isfinite(peak_memory).any())
                else None
            ),
            'peak_cuda_memory_mb_max': (
                float(np.nanmax(peak_memory))
                if bool(np.isfinite(peak_memory).any())
                else None
            ),
            'n':               int(len(psnr_rss)),
        }
        volume_psnr_rss = defaultdict(list)
        volume_ssim_rss = defaultdict(list)
        volume_nmse_rss = defaultdict(list)
        volume_psnr_sense = defaultdict(list)
        volume_ssim_sense = defaultdict(list)
        for row in all_results:
            if m in row['methods']:
                volume_psnr_rss[row['volume']].append(
                    row['methods'][m].get(
                        'canonical_psnr_rss',
                        row['methods'][m]['psnr_rss'],
                    ))
                volume_ssim_rss[row['volume']].append(
                    row['methods'][m].get(
                        'canonical_ssim_rss',
                        row['methods'][m]['ssim_rss'],
                    ))
                volume_nmse_rss[row['volume']].append(
                    row['methods'][m].get(
                        'canonical_nmse_rss',
                        row['methods'][m]['nmse_rss'],
                    ))
                volume_psnr_sense[row['volume']].append(
                    row['methods'][m].get(
                        'canonical_psnr_sense',
                        row['methods'][m]['psnr_sense'],
                    ))
                volume_ssim_sense[row['volume']].append(
                    row['methods'][m].get(
                        'canonical_ssim_sense',
                        row['methods'][m]['ssim_sense'],
                    ))
        psnr_rss_volume_means = np.array(
            [np.mean(values) for values in volume_psnr_rss.values()])
        ssim_rss_volume_means = np.array(
            [np.mean(values) for values in volume_ssim_rss.values()])
        nmse_rss_volume_means = np.array(
            [np.mean(values) for values in volume_nmse_rss.values()])
        psnr_sense_volume_means = np.array(
            [np.mean(values) for values in volume_psnr_sense.values()])
        ssim_sense_volume_means = np.array(
            [np.mean(values) for values in volume_ssim_sense.values()])
        summary[m].update({
            'selection_psnr_rss_volume_mean': float(psnr_rss_volume_means.mean()),
            'selection_psnr_rss_volume_std': float(psnr_rss_volume_means.std()),
            'selection_ssim_rss_volume_mean': float(ssim_rss_volume_means.mean()),
            'selection_ssim_rss_volume_std': float(ssim_rss_volume_means.std()),
            'selection_nmse_rss_volume_mean': float(nmse_rss_volume_means.mean()),
            'selection_nmse_rss_volume_std': float(nmse_rss_volume_means.std()),
            'psnr_sense_volume_mean': float(psnr_sense_volume_means.mean()),
            'psnr_sense_volume_std': float(psnr_sense_volume_means.std()),
            'ssim_sense_volume_mean': float(ssim_sense_volume_means.mean()),
            'ssim_sense_volume_std': float(ssim_sense_volume_means.std()),
            'n_volumes': int(len(psnr_rss_volume_means)),
        })
        print(f'  {m:14s}  budget={int(budget.mean()):3d} cols  '
              f'NFE={np.nanmean(nfe):.1f}  '
              f'PSNR_RSS={psnr_rss.mean():.3f}±{psnr_rss.std():.3f}  '
              f'PSNR_SENSE={psnr_sense.mean():.3f}±{psnr_sense.std():.3f}  '
              f'SSIM_RSS={ssim_rss.mean():.4f}  SSIM_SENSE={ssim_sense.mean():.4f}  '
              f'SELECTION_RSS_PSNR={selection_psnr_rss.mean():.3f}')
        # PV diagnostics: K-divergence vs PiGDM and angular fraction of P
        kdiv = [r['methods'][m].get('K_div_mean')
                for r in all_results if m in r['methods']
                and 'K_div_mean' in r['methods'][m]]
        pang = [r['methods'][m].get('P_ang_frac_mean')
                for r in all_results if m in r['methods']
                and 'P_ang_frac_mean' in r['methods'][m]]
        if kdiv and pang:
            kd = np.array(kdiv); pa = np.array(pang)
            print(f'                  PV-diag: K_div(vs PiGDM)={kd.mean():.4f}±{kd.std():.4f}  '
                  f'P_ang_frac={pa.mean():.4f}±{pa.std():.4f}  '
                  f'(0 ⇒ gate ≡ PiGDM / P purely radial)')
            summary[m]['K_div_mean']      = float(kd.mean())
            summary[m]['P_ang_frac_mean'] = float(pa.mean())

    # Cluster-aware primary inference on RSS-target PSNR: first average paired
    # slice differences
    # within each volume, then apply Wilcoxon across independent volumes.
    # Per-slice p-values would pseudoreplicate adjacent anatomy.
    try:
        from scipy.stats import wilcoxon
        candidates = [
            'operator_diag_active',
            'sketch_marginal_active',
            'sketch_logdet_active',
            'pv_active',
            'energy_active',
            'oracle_active',
        ]
        baselines  = ['static_match', 'random_adaptive', 'equi_adaptive']
        for cand in candidates:
            if cand not in summary:
                continue
            for ref in baselines + [c for c in candidates if c != cand]:
                if ref not in summary:
                    continue
                paired_by_volume = defaultdict(list)
                for row in all_results:
                    if cand in row['methods'] and ref in row['methods']:
                        paired_by_volume[row['volume']].append(
                            row['methods'][cand].get(
                                'canonical_psnr_rss',
                                row['methods'][cand]['psnr_rss'],
                            )
                            - row['methods'][ref].get(
                                'canonical_psnr_rss',
                                row['methods'][ref]['psnr_rss'],
                            ))
                if not paired_by_volume:
                    continue
                d = np.array([
                    np.mean(values) for values in paired_by_volume.values()
                ])
                try:
                    stat, pval = wilcoxon(d, alternative='greater')
                except ValueError:
                    pval = float('nan')
                ci_low, ci_high = _paired_bootstrap_mean_ci(
                    d,
                    seed=args.seed + 91,
                )
                print(f'  {cand:14s} − {ref:15s}  ΔRSS={d.mean():+.3f}±{d.std():.3f} dB '
                      f'(n_volumes={len(d)}, Wilcoxon p={pval:.2e})')
                summary[f'_{cand}_vs_{ref}'] = {
                    'delta_selection_psnr_rss_mean': float(d.mean()),
                    'delta_selection_psnr_rss_std':  float(d.std()),
                    'paired_bootstrap_95ci': [ci_low, ci_high],
                    'wilcoxon_p_greater':    float(pval),
                    'n_volumes':             int(len(d)),
                }
    except ImportError:
        pass

    out_json = os.path.join(args.output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'status': 'CORRECTED_RUN_UNREVIEWED',
            'config': vars(args),
            'protocol': {
                'primary_selection_metric': (
                    'system reconstruction'
                    if args.skip_canonical_replay
                    else 'canonical single-trajectory replay of the final mask'
                ),
                'system_and_selection_metrics_separated': True,
                'dataset_revision_complete': (
                    args.dataset_revision != 'unspecified'
                ),
                'input_hashes_complete': not args.skip_file_hashes,
            },
            'environment': {
                'torch_version': torch.__version__,
                'cuda_version': torch.version.cuda,
                'device': str(device),
                'device_name': (
                    torch.cuda.get_device_name(device)
                    if device.type == 'cuda'
                    else str(device)
                ),
            },
            'git_commit': subprocess_git_commit(),
            'volume_manifest': volume_manifest,
            'per_slice': all_results,
            'summary': summary,
        }, f, indent=2)
    print(f'\nSaved → {out_json}')


if __name__ == '__main__':
    main()
