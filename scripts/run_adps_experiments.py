"""Run ADPS experiments on brain and knee data to match existing DPS/ΠGDM/FA-KGD results."""

import sys, time, json
sys.path.insert(0, '.')
import h5py, torch, numpy as np
from skimage.metrics import structural_similarity as ssim_fn
from src.samplers.mri_forward import fft2c, ifft2c, create_mask
from src.samplers.adps import run_adps
from src.models.edm_loader import load_edm_model, EDMDenoiser


def make_freq_noise(H, W, sigma_base=0.001, beta_noise=5.0):
    cy, cx = H // 2, W // 2
    gy, gx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    r = torch.sqrt((gy - cy).float() ** 2 + (gx - cx).float() ** 2)
    return sigma_base * (1 + beta_noise * (r / r.max()) ** 2)


def add_noise(y, sq):
    s = sq.sqrt()
    return y + s * (torch.randn_like(y.real) + 1j * torch.randn_like(y.real)) / np.sqrt(2)


def load_brain_slice(vol_path, sl, H=384, W=320):
    with h5py.File(vol_path, 'r') as f:
        rss = f['reconstruction_rss'][sl]
        img = torch.from_numpy(rss).to(torch.complex64)
    sh = (img.shape[0] - H) // 2
    sw = (img.shape[1] - W) // 2
    return img[sh:sh+H, sw:sw+W]


def load_knee_slice(vol_path, sl, H=384, W=320):
    with h5py.File(vol_path, 'r') as f:
        kspace = torch.from_numpy(f['kspace'][sl])
    img = ifft2c(kspace)
    sh = (img.shape[0] - H) // 2
    sw = (img.shape[1] - W) // 2
    return img[sh:sh+H, sw:sw+W]


def run_experiment(experiment_name, vol_path, slices, accel, load_fn, model, sigma_schedule, l_ss=10.0):
    H, W = 384, 320
    results = []
    
    for sl in slices:
        x_gt = load_fn(vol_path, sl, H, W)
        scale = x_gt.abs().max().item()
        x_gt = x_gt / scale
        
        mask_1d = create_mask(W, center_fraction=0.08, acceleration=accel, seed=42)
        mask = mask_1d.expand(H, -1)
        y_full = fft2c(x_gt)
        true_sigma_sq = make_freq_noise(H, W)
        y_noisy = add_noise(y_full, true_sigma_sq)
        y = mask * y_noisy
        
        t0 = time.time()
        res = run_adps(
            y=y, mask=mask, sigma_schedule=sigma_schedule,
            denoiser_fn=model, step_size=l_ss, s_churn=0.0,
            x_gt=x_gt, seed=0,
        )
        elapsed = time.time() - t0
        psnr = res['psnr_trajectory'][-1]
        gt_mag = x_gt.abs().cpu().numpy()
        recon_mag = res['recon'].abs().cpu().numpy()
        dr = gt_mag.max() - gt_mag.min()
        ssim_val = ssim_fn(gt_mag, recon_mag, data_range=dr)
        print(f'  {experiment_name} sl{sl}: ADPS PSNR={psnr:.2f} SSIM={ssim_val:.4f} ({elapsed:.0f}s)')
        results.append({'slice': sl, 'psnr': psnr, 'ssim': float(ssim_val)})
    
    psnrs = [r['psnr'] for r in results]
    ssims = [r['ssim'] for r in results]
    print(f'  {experiment_name} MEAN: PSNR={np.mean(psnrs):.2f}±{np.std(psnrs):.2f} SSIM={np.mean(ssims):.4f}±{np.std(ssims):.4f}')
    return results


def main():
    print("Loading EDM model...")
    net = load_edm_model('checkpoints/edm/supervised_R=1', device='cpu')
    model = EDMDenoiser(net, device='cpu')
    
    # EDM schedule T=20
    sigma_max, sigma_min, rho = 10.0, 0.002, 7
    step_indices = torch.arange(20, dtype=torch.float64)
    sigma_schedule = (sigma_max**(1/rho) + step_indices/19 * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    sigma_schedule = sigma_schedule.float()
    
    all_results = {}
    
    # Brain R=4 T=20 (same 5 slices as edm_brain_R4_T20_dps)
    print("\n=== Brain R=4 T=20 ===")
    brain_vol = 'data/brain_val/file_brain_AXT2_200_2000093.h5'
    all_results['brain_R4_T20'] = run_experiment(
        'Brain R4', brain_vol, range(6, 11), accel=4,
        load_fn=load_brain_slice, model=model, sigma_schedule=sigma_schedule,
    )
    
    # Brain R=4 T=50
    print("\n=== Brain R=4 T=50 ===")
    step_indices_50 = torch.arange(50, dtype=torch.float64)
    sigma_schedule_50 = (sigma_max**(1/rho) + step_indices_50/49 * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    sigma_schedule_50 = sigma_schedule_50.float()
    all_results['brain_R4_T50'] = run_experiment(
        'Brain R4 T50', brain_vol, range(6, 11), accel=4,
        load_fn=load_brain_slice, model=model, sigma_schedule=sigma_schedule_50,
    )
    
    # Knee R=4 T=20 (same 5 slices as edm_knee_R4_T20_dps)
    print("\n=== Knee R=4 T=20 ===")
    knee_vol = 'data/singlecoil_test/file1000022.h5'
    all_results['knee_R4_T20'] = run_experiment(
        'Knee R4', knee_vol, range(16, 21), accel=4,
        load_fn=load_knee_slice, model=model, sigma_schedule=sigma_schedule,
    )
    
    # Knee R=8 T=20 (same 5 slices)
    print("\n=== Knee R=8 T=20 ===")
    all_results['knee_R8_T20'] = run_experiment(
        'Knee R8', knee_vol, range(16, 21), accel=8,
        load_fn=load_knee_slice, model=model, sigma_schedule=sigma_schedule,
    )
    
    # Save results
    with open('outputs/adps_all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to outputs/adps_all_results.json")


if __name__ == '__main__':
    main()
