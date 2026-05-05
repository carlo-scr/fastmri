"""Multi-coil reconstruction with PV-gated FA-KGD + active line acquisition.

Sister script to ``scripts/reconstruct.py``. Operates on raw multi-coil
k-space from fastMRI brain (and similar), and supports the full
PV/active-sampling family.

Methods (``--methods`` list, any subset):
  pigdm_mc     : multi-coil ΠGDM with per-coil scalar noise
  fakgd_mc     : multi-coil FA-KGD+FPDC (per-frequency Kalman gain)
  pv           : (A) PV-gated FA-KGD only
  pv_active    : (A) + (D) active line acquisition
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
import json
import os
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
from src.samplers.multicoil_pv import run_fakgd_mc_pv
from src.models.edm_loader import load_edm_model, EDMDenoiser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
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


def _static_matched_mask(W: int, H: int, n_total_cols: int, cf: float,
                         seed: int, device: torch.device) -> torch.Tensor:
    """Build a 2D mask (H, W) with `n_total_cols` columns ON, including a
    centred ACS region of width round(cf*W) (rounded up to even)."""
    nacs = int(round(cf * W))
    if nacs % 2:
        nacs += 1
    nacs = min(nacs, n_total_cols)
    n_rand = max(0, n_total_cols - nacs)
    acs_set = set(range(W // 2 - nacs // 2, W // 2 + nacs // 2))
    cand = list(set(range(W)) - acs_set)
    rng = np.random.RandomState(seed)
    sel = rng.choice(cand, size=min(n_rand, len(cand)), replace=False) if n_rand else []
    m = np.zeros(W, dtype=np.float32)
    for c in acs_set:
        m[c] = 1
    for c in sel:
        m[int(c)] = 1
    return torch.from_numpy(m).expand(H, -1).to(device)


def _add_kspace_noise(mc_k: torch.Tensor, sig_c: torch.Tensor,
                      seed: int) -> torch.Tensor:
    """Add white per-coil Gaussian noise (variance sig_c[c]) to mc_k."""
    Nc = mc_k.shape[0]
    g = torch.Generator(device='cpu').manual_seed(seed)
    nz = (torch.randn(mc_k.shape, generator=g)
          + 1j * torch.randn(mc_k.shape, generator=g)) / np.sqrt(2)
    return mc_k + (sig_c.sqrt().view(Nc, 1, 1) * nz).to(mc_k.dtype)


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


# ---------------------------------------------------------------------------
# Per-slice runner
# ---------------------------------------------------------------------------

def run_slice(args, denoiser, sigma_schedule, mc_k_clean, rss_gt,
              file_seed: int, slice_idx: int, device: torch.device) -> dict:
    """Reconstruct one slice with all requested methods. Returns metrics dict."""
    Nc, H, W = mc_k_clean.shape

    # Normalise so that |RSS| has max 1.0 (consistent with EDM training)
    scale = float(rss_gt.max()).__pow__(1.0)
    if scale < 1e-12:
        scale = 1.0
    mc_k = mc_k_clean / scale
    rss = (rss_gt / scale).to(device)

    # Per-coil noise variance (estimated from data corners), shared across all methods
    sig_c = estimate_noise_per_coil(mc_k).to(device)
    # If the user asked for a synthetic noise floor, inflate sig_c to that scalar
    if args.noise_scale is not None and args.noise_scale > 0:
        floor = float(args.noise_scale) ** 2
        sig_c = torch.full_like(sig_c, floor)
    # Add measurement noise (deterministic per slice)
    noise_seed = args.noise_seed + file_seed + slice_idx * 7919
    mc_kn = _add_kspace_noise(mc_k, sig_c, seed=noise_seed).to(device)
    mc_k = mc_k.to(device)

    # R-mask (1D, expanded to 2D)
    mask_R = create_mask(W, args.center_fraction, args.acceleration,
                         seed=args.mask_seed).expand(H, -1).to(device)
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
        'n_R_cols': n_R_cols,
        'methods': {},
    }

    def _record(name, recon, total_cols, runtime):
        rec_mag = recon.abs() if torch.is_complex(recon) else recon
        rec_mag = rec_mag.to(device)
        out['methods'][name] = {
            'psnr_rss':    _psnr(rss, rec_mag, eval_size=args.eval_size),
            'psnr_sense':  _psnr(sense_oracle, rec_mag, eval_size=args.eval_size),
            'ssim_rss':    _ssim(rss, rec_mag, eval_size=args.eval_size),
            'ssim_sense':  _ssim(sense_oracle, rec_mag, eval_size=args.eval_size),
            'budget_cols': int(total_cols),
            'runtime_s':   float(runtime),
        }

    y_R = mask_R * mc_kn

    # ---- pigdm_mc ----
    if 'pigdm_mc' in args.methods:
        t0 = time.time()
        r = run_pigdm_mc(y_mc=y_R, sense_op=sense_op,
                         sigma_schedule=sigma_schedule,
                         denoiser_fn=denoiser, sigma_y=None, seed=args.seed)
        _record('pigdm_mc', r['recon'], n_R_cols, time.time() - t0)

    # ---- fakgd_mc (FPDC, no PV) ----
    if 'fakgd_mc' in args.methods:
        t0 = time.time()
        r = run_fakgd_mc(y_mc=y_R, sense_op=sense_op,
                         sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                         sigma_i_sq_init=sigma_i_sq, r_acs=r_acs, r_max=r_max,
                         beta_fpdc=args.beta_fpdc, alpha_ema=args.alpha_ema,
                         gamma=args.gamma, m_step_mode=args.m_step_mode,
                         seed=args.seed)
        _record('fakgd_mc', r['recon'], n_R_cols, time.time() - t0)

    # ---- pv (PV-gated only) ----
    if 'pv' in args.methods:
        t0 = time.time()
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=True, refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            active_lines=0, seed=args.seed)
        _record('pv', r['recon'], n_R_cols, time.time() - t0)

    # ---- pv_active (PV + active acquisition) ----
    if 'pv_active' in args.methods and args.active_lines > 0:
        t0 = time.time()
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=True, refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='pv',
                            y_mc_full=mc_kn, seed=args.seed)
        _record('pv_active', r['recon'], n_R_cols + args.active_lines,
                time.time() - t0)

    # ---- random_adaptive (mid-diffusion uniform-random acquisition) ----
    # Same multi-round schedule and budget as PVAS; columns picked uniformly
    # at random instead of by PV score. Isolates the value of variance-guided
    # selection (vs. "any mid-diffusion adaptation").
    if 'random_adaptive' in args.methods and args.active_lines > 0:
        t0 = time.time()
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=True, refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='random',
                            y_mc_full=mc_kn, seed=args.seed)
        _record('random_adaptive', r['recon'], n_R_cols + args.active_lines,
                time.time() - t0)

    # ---- equi_adaptive (mid-diffusion gap-filling acquisition) ----
    # Deterministic non-PV baseline: greedily fill the largest column gaps.
    if 'equi_adaptive' in args.methods and args.active_lines > 0:
        t0 = time.time()
        r = run_fakgd_mc_pv(y_mc=y_R, sense_op=sense_op,
                            sigma_schedule=sigma_schedule, denoiser_fn=denoiser,
                            sigma_i_sq_init=sigma_i_sq,
                            use_pv_gate=True, refine_sens=False,
                            pv_eps_probe=args.pv_eps_probe,
                            pv_centered=args.pv_centered,
                            active_lines=args.active_lines,
                            active_rounds=args.active_rounds,
                            active_after_frac=args.active_after_frac,
                            active_until_frac=args.active_until_frac,
                            active_score='equispaced',
                            y_mc_full=mc_kn, seed=args.seed)
        _record('equi_adaptive', r['recon'], n_R_cols + args.active_lines,
                time.time() - t0)

    # ---- static_match (matched-budget static-random baseline) ----
    if 'static_match' in args.methods and args.active_lines > 0:
        budget = n_R_cols + args.active_lines
        mask_static = _static_matched_mask(W, H, budget, args.center_fraction,
                                           seed=args.mask_seed, device=device)
        op_static = MultiCoilSENSE(mask_static, sens, sig_c).to(device)
        t0 = time.time()
        r = run_pigdm_mc(y_mc=mask_static * mc_kn, sense_op=op_static,
                         sigma_schedule=sigma_schedule,
                         denoiser_fn=denoiser, sigma_y=None, seed=args.seed)
        _record('static_match', r['recon'], budget, time.time() - t0)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--data_path', type=str, required=True)
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
    p.add_argument('--noise_scale', type=float, default=None,
                   help='If set, use uniform per-coil σ = noise_scale; '
                        'otherwise estimate from data corners.')
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

    # Active sampling params
    p.add_argument('--active_lines', type=int, default=0)
    p.add_argument('--active_rounds', type=int, default=1)
    p.add_argument('--active_after_frac', type=float, default=0.5)
    p.add_argument('--active_until_frac', type=float, default=0.85)

    # Methods
    p.add_argument('--methods', type=str, nargs='+',
                   default=['pigdm_mc', 'fakgd_mc', 'pv', 'pv_active',
                            'random_adaptive', 'static_match'],
                   choices=['pigdm_mc', 'fakgd_mc', 'pv', 'pv_active',
                            'random_adaptive', 'equi_adaptive', 'static_match'])

    # Output
    p.add_argument('--output_dir', type=str, required=True)

    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == 'cpu' else 'cpu')

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

    for fi, fp in enumerate(files):
        fname = Path(fp).name
        file_seed = int.from_bytes(
            __import__('hashlib').blake2b(fname.encode(), digest_size=4).digest(),
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
            res.update({'volume': fname, 'slice': si})
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
        budget     = np.array([r['methods'][m]['budget_cols'] for r in all_results if m in r['methods']])
        summary[m] = {
            'psnr_rss_mean':   float(psnr_rss.mean()),   'psnr_rss_std':   float(psnr_rss.std()),
            'psnr_sense_mean': float(psnr_sense.mean()), 'psnr_sense_std': float(psnr_sense.std()),
            'ssim_rss_mean':   float(ssim_rss.mean()),   'ssim_rss_std':   float(ssim_rss.std()),
            'ssim_sense_mean': float(ssim_sense.mean()), 'ssim_sense_std': float(ssim_sense.std()),
            'budget_cols':     int(budget.mean()),
            'n':               int(len(psnr_rss)),
        }
        print(f'  {m:14s}  budget={int(budget.mean()):3d} cols  '
              f'PSNR_RSS={psnr_rss.mean():.3f}±{psnr_rss.std():.3f}  '
              f'PSNR_SENSE={psnr_sense.mean():.3f}±{psnr_sense.std():.3f}  '
              f'SSIM_RSS={ssim_rss.mean():.4f}  SSIM_SENSE={ssim_sense.mean():.4f}')

    # Paired Wilcoxon for pv_active vs each non-PV adaptive/static baseline
    if 'pv_active' in summary:
        try:
            from scipy.stats import wilcoxon
            for ref in ('static_match', 'random_adaptive', 'equi_adaptive'):
                if ref not in summary:
                    continue
                paired = [(r['methods']['pv_active']['psnr_sense'],
                           r['methods'][ref]['psnr_sense'])
                          for r in all_results
                          if 'pv_active' in r['methods'] and ref in r['methods']]
                if not paired:
                    continue
                a = np.array([p[0] for p in paired]); b = np.array([p[1] for p in paired])
                d = a - b
                try:
                    stat, pval = wilcoxon(d, alternative='greater')
                except ValueError:
                    pval = float('nan')
                print(f'\n  pv_active − {ref:15s}  ΔSENSE={d.mean():+.3f}±{d.std():.3f} dB '
                      f'(n={len(d)}, Wilcoxon p={pval:.2e})')
                summary[f'_pv_vs_{ref}'] = {
                    'delta_psnr_sense_mean': float(d.mean()),
                    'delta_psnr_sense_std':  float(d.std()),
                    'wilcoxon_p_greater':    float(pval),
                    'n':                     int(len(d)),
                }
        except ImportError:
            pass

    out_json = os.path.join(args.output_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({
            'config': vars(args),
            'per_slice': all_results,
            'summary': summary,
        }, f, indent=2)
    print(f'\nSaved → {out_json}')


if __name__ == '__main__':
    main()
