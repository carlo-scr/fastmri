"""Multi-slice/volume validation of PV-active sampling vs static baselines."""
import sys, time, glob
import numpy as np
import torch
import h5py
sys.path.insert(0, '.')
from src.samplers.mri_forward import fft2c, ifft2c, create_mask
from src.samplers.schedules import edm_sigma_schedule
from src.samplers.sense import (
    estimate_sens_maps_lowres, estimate_noise_per_coil,
    MultiCoilSENSE, sense_combine,
)
from src.samplers.multicoil import run_pigdm_mc
from src.samplers.multicoil_pv import run_fakgd_mc_pv
from src.models.edm_loader import load_edm_model, EDMDenoiser


def cc(x, h, w):
    H, W = x.shape[-2:]
    sh = (H - h) // 2
    sw = (W - w) // 2
    return x[..., sh:sh + h, sw:sw + w]


def psnr(g, r):
    H = min(g.shape[-2], 320)
    W = min(g.shape[-1], 320)
    g = cc(g, H, W); r = cc(r, H, W)
    dr = g.max(); m = ((g - r) ** 2).mean()
    return 100.0 if m < 1e-12 else (10 * torch.log10(dr ** 2 / m)).item()


def static_mask(W, H, n_total, cf, seed, device):
    nacs = int(round(cf * W))
    nacs = nacs if nacs % 2 == 0 else nacs + 1
    n_rand = n_total - nacs
    rng = np.random.RandomState(seed)
    cand = list(set(range(W)) - set(range(W // 2 - nacs // 2, W // 2 + nacs // 2)))
    sel = rng.choice(cand, size=n_rand, replace=False)
    m = np.zeros(W, dtype=np.float32)
    m[W // 2 - nacs // 2:W // 2 + nacs // 2] = 1
    m[sel] = 1
    return torch.from_numpy(m).expand(H, -1).to(device)


def main():
    device = torch.device('cpu')
    net = load_edm_model('checkpoints/edm/supervised_R=1', method='auto', device='cpu')
    denoiser = EDMDenoiser(net, device='cpu')
    sigma = edm_sigma_schedule(20).to(device)

    files = sorted(glob.glob('data/brain_val/file_brain_AXT2_*.h5'))[:2]
    slices = [6, 8, 10]

    results = {k: [] for k in ['static_R4', 'pv_a8', 'static_m8', 'pv_a16', 'static_m16']}

    for fp in files:
        with h5py.File(fp, 'r') as h:
            ks_all = h['kspace'][:]
        for sl in slices:
            t_slice = time.time()
            ks = torch.from_numpy(ks_all[sl])
            ci = cc(ifft2c(ks), 384, 320)
            rss = torch.sqrt((ci.abs() ** 2).sum(0))
            scale = float(rss.max())
            ci = ci / scale
            rss = rss / scale
            mc_k = fft2c(ci)
            Nc, H, W = mc_k.shape

            sigc = estimate_noise_per_coil(mc_k).to(device)
            gen = torch.Generator(device='cpu').manual_seed(8)
            nz = (torch.randn(mc_k.shape, generator=gen)
                  + 1j * torch.randn(mc_k.shape, generator=gen)) / np.sqrt(2)
            mc_kn = mc_k + (sigc.sqrt().view(Nc, 1, 1) * nz).to(mc_k.dtype)

            mask4 = create_mask(W, 0.08, 4, seed=42).expand(H, -1).to(device)
            sens = estimate_sens_maps_lowres(mc_kn, center_fraction=0.08).to(device)
            op = MultiCoilSENSE(mask4, sens, sigc).to(device)
            so = sense_combine(ifft2c(mc_k), sens)
            sigi = torch.full((H, W), float(sigc.mean()), device=device)
            n_R4 = int(mask4[0].sum())

            # static R=4
            r = run_pigdm_mc(y_mc=mask4 * mc_kn, sense_op=op,
                             sigma_schedule=sigma, denoiser_fn=denoiser,
                             sigma_y=None, seed=0)
            results['static_R4'].append((psnr(rss, r['recon'].abs()),
                                         psnr(so.abs(), r['recon'].abs())))

            # PV + active 8
            r = run_fakgd_mc_pv(y_mc=mask4 * mc_kn, sense_op=op,
                                sigma_schedule=sigma, denoiser_fn=denoiser,
                                sigma_i_sq_init=sigi, use_pv_gate=True,
                                refine_sens=False, active_lines=8,
                                y_mc_full=mc_kn, seed=0)
            results['pv_a8'].append((psnr(rss, r['recon'].abs()),
                                     psnr(so.abs(), r['recon'].abs())))

            # static matched to active 8
            mask_st = static_mask(W, H, n_R4 + 8, 0.08, seed=42, device=device)
            op_st = MultiCoilSENSE(mask_st, sens, sigc).to(device)
            r = run_pigdm_mc(y_mc=mask_st * mc_kn, sense_op=op_st,
                             sigma_schedule=sigma, denoiser_fn=denoiser,
                             sigma_y=None, seed=0)
            results['static_m8'].append((psnr(rss, r['recon'].abs()),
                                         psnr(so.abs(), r['recon'].abs())))

            # PV + active 16
            r = run_fakgd_mc_pv(y_mc=mask4 * mc_kn, sense_op=op,
                                sigma_schedule=sigma, denoiser_fn=denoiser,
                                sigma_i_sq_init=sigi, use_pv_gate=True,
                                refine_sens=False, active_lines=16,
                                y_mc_full=mc_kn, seed=0)
            results['pv_a16'].append((psnr(rss, r['recon'].abs()),
                                      psnr(so.abs(), r['recon'].abs())))

            # static matched to active 16
            mask_st = static_mask(W, H, n_R4 + 16, 0.08, seed=42, device=device)
            op_st = MultiCoilSENSE(mask_st, sens, sigc).to(device)
            r = run_pigdm_mc(y_mc=mask_st * mc_kn, sense_op=op_st,
                             sigma_schedule=sigma, denoiser_fn=denoiser,
                             sigma_y=None, seed=0)
            results['static_m16'].append((psnr(rss, r['recon'].abs()),
                                          psnr(so.abs(), r['recon'].abs())))

            print(f'{fp.split("/")[-1]} sl={sl} ({time.time()-t_slice:.0f}s):  '
                  f'R4={results["static_R4"][-1][1]:.2f}  '
                  f'pv_a8={results["pv_a8"][-1][1]:.2f}  '
                  f'st_m8={results["static_m8"][-1][1]:.2f}  '
                  f'pv_a16={results["pv_a16"][-1][1]:.2f}  '
                  f'st_m16={results["static_m16"][-1][1]:.2f}',
                  flush=True)

    print(f'\n=== AGGREGATE ({len(results["static_R4"])} slices, SENSE-PSNR) ===')
    for k in ['static_R4', 'pv_a8', 'static_m8', 'pv_a16', 'static_m16']:
        a = np.array(results[k])
        print(f'  {k:12s} RSS={a[:,0].mean():.3f}±{a[:,0].std():.3f}  '
              f'SENSE={a[:,1].mean():.3f}±{a[:,1].std():.3f}')

    a = np.array(results['pv_a8'])
    b = np.array(results['static_m8'])
    print(f'\n  pv_a8  vs static_m8: ΔSENSE = {a[:,1].mean()-b[:,1].mean():+.3f} dB')
    a = np.array(results['pv_a16'])
    b = np.array(results['static_m16'])
    print(f'  pv_a16 vs static_m16: ΔSENSE = {a[:,1].mean()-b[:,1].mean():+.3f} dB')


if __name__ == '__main__':
    main()
