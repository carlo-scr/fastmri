"""Posterior-Variance-Gated FA-KGD with optional in-loop SENSE refinement
and active line acquisition. The three new components on top of
`run_fakgd_mc`:

  (A) Posterior-Variance-Gated DC.   The Kalman gain is

          K_{c,i}(t) = P_i(t) / (P_i(t) + σ_{c,i}^2)

      with P_i(t) the Tweedie second-moment estimate of the per-frequency
      posterior variance (see `posterior_var.py`). Replaces FPDC's
      hard-radius gate with a soft, theory-derived per-frequency gain.
      At low t (early sampling) P_i is small at high frequencies → small
      gain → automatic deferral, exactly mirroring FPDC. At high t (late
      sampling) P_i grows so all frequencies receive their measurement.

  (B) In-loop SENSE refinement.  Every `sens_refresh_every` steps, refine
      the sensitivity maps using the current iterate's magnitude:

          S_c <- LowPass( IFFT(M*y_c) / x̂_0 )

      enforced smooth via a Gaussian low-pass in image domain. This is
      the diffusion analogue of JSENSE — the first time it has been used
      inside a training-free MR-diffusion sampler.

  (D) Active line selection.  After `active_warmup_frac * T` steps, query
      `num_active_lines` extra k-space columns chosen as those that
      maximise expected information gain, scored by

          score(col) = sum_r∈col P_r(t) / σ_{c,r}^2.

      We then "acquire" them by setting `mask[col]=1` and treating the
      ground-truth k-space at those lines as the new measurements (since
      this is a retrospective study). Optional via `active_*` args.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c, build_radius_grid
from .sense import MultiCoilSENSE
from .posterior_var import estimate_posterior_variance_kspace


def _complex_randn(H, W, device):
    return (torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)) / np.sqrt(2)


def _psnr(x_gt, x_recon):
    gt = x_gt.abs(); rc = x_recon.abs()
    H = min(gt.shape[-2], 320); W = min(gt.shape[-1], 320)
    sh = (gt.shape[-2] - H) // 2; sw = (gt.shape[-1] - W) // 2
    gt = gt[..., sh:sh+H, sw:sw+W]; rc = rc[..., sh:sh+H, sw:sw+W]
    dr = gt.max(); mse = ((gt - rc) ** 2).mean()
    return 100.0 if mse < 1e-12 else (10 * torch.log10(dr ** 2 / mse)).item()


def _gaussian_lowpass_2d(H, W, cutoff_frac, device):
    """Gaussian LP filter with cutoff at cutoff_frac * Nyquist."""
    fy = torch.arange(H, device=device).float() - H // 2
    fx = torch.arange(W, device=device).float() - W // 2
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(gy ** 2 + gx ** 2)
    sigma = max(2.0, cutoff_frac * min(H, W) / 2.0)
    return torch.exp(-0.5 * (r / sigma) ** 2)


def refine_sens_maps_inloop(
    mc_y_obs: torch.Tensor,           # (Nc,H,W) sampled k-space (M*y_c)
    mask: torch.Tensor,               # (H,W)
    x_hat: torch.Tensor,              # (H,W) current SENSE-combined estimate
    sens_prev: torch.Tensor,          # (Nc,H,W) previous map (used as prior)
    cutoff_frac: float = 0.10,        # LP filter cutoff (relative)
    blend: float = 0.3,               # EMA between sens_prev and refined
    eps: float = 1e-3,
    magnitude_only: bool = True,
) -> torch.Tensor:
    """One JSENSE-style refinement step.

    Magnitude-only mode (default): only refine the *amplitude* of S_c
    using the magnitude ratio of coil images to the current iterate
    magnitude, keeping the prior phase. This is robust at intermediate
    diffusion steps where the iterate's phase is still noisy.

    Full-complex mode (`magnitude_only=False`): the standard JSENSE
    division S_c = LP(coil_img / x̂_0). Use only when iterate is clean.
    """
    Nc, H, W = mc_y_obs.shape
    device = mc_y_obs.device
    coil_imgs = ifft2c(mc_y_obs)       # (Nc,H,W) complex

    x_mag = x_hat.abs()
    x_floor = eps * x_mag.max().clamp(min=1e-8)

    if magnitude_only:
        # Estimate |S_c| from |coil_img_c| / |x̂_0|, keep phase from sens_prev
        sens_mag_raw = coil_imgs.abs() / (x_mag.unsqueeze(0) + x_floor)
        # Smooth magnitude in k-space
        lp = _gaussian_lowpass_2d(H, W, cutoff_frac, device)
        sens_mag_smooth = ifft2c(
            fft2c(sens_mag_raw.to(sens_prev.dtype)) * lp.unsqueeze(0)
        ).real.clamp(min=0.0)
        # Foreground mask
        fg = (x_mag > 0.02 * x_mag.max()).float()
        sens_mag_smooth = sens_mag_smooth * fg.unsqueeze(0)
        # Combine: keep prior phase, update magnitude
        prev_phase = sens_prev / (sens_prev.abs() + 1e-12)
        sens_refined = sens_mag_smooth * prev_phase
    else:
        # Full complex JSENSE: 1/x = conj(x)/|x|^2
        inv_x = x_hat.conj() / (x_mag ** 2 + x_floor ** 2)
        sens_raw = coil_imgs * inv_x.unsqueeze(0)
        lp = _gaussian_lowpass_2d(H, W, cutoff_frac, device)
        sens_refined = ifft2c(fft2c(sens_raw) * lp.unsqueeze(0))
        fg = (x_mag > 0.02 * x_mag.max()).to(sens_refined.dtype)
        sens_refined = sens_refined * fg.unsqueeze(0)

    return (1 - blend) * sens_prev + blend * sens_refined


def run_fakgd_mc_pv(
    y_mc: torch.Tensor,
    sense_op: MultiCoilSENSE,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    # noise model
    sigma_i_sq_init: torch.Tensor,
    eps: float = 1e-8,
    # (A) posterior-variance gating
    use_pv_gate: bool = True,
    pv_eps_probe: float = 5e-2,
    pv_radial_smooth: bool = True,
    pv_radial_degree: int = 4,
    pv_centered: bool = True,
    pv_normalize: bool = True,            # rescale P̂ so its mean ≈ σ_t²
    pv_n_probes: int = 1,
    # (B) in-loop sens refinement
    refine_sens: bool = True,
    sens_refresh_every: int = 4,
    sens_warmup_frac: float = 0.3,        # only refine after this much of T
    sens_blend: float = 0.5,
    sens_lp_cutoff: float = 0.10,
    # (D) active sampling
    active_lines: int = 0,                # total extra columns to query
    active_rounds: int = 1,               # split across this many timesteps (>=1)
    active_after_frac: float = 0.5,       # first round at this fraction of T
    active_until_frac: float = 0.85,      # last round at this fraction of T
    active_score: str = "pv",             # "pv" | "random" | "equispaced"
    y_mc_full: Optional[torch.Tensor] = None,   # full k-space (Nc,H,W) for retrospective query
    x_gt: Optional[torch.Tensor] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
) -> dict:
    """Posterior-Variance-Gated multi-coil FA-KGD with optional in-loop
    sens refinement and active line acquisition.
    """
    torch.manual_seed(seed)
    Nc, H, W = y_mc.shape
    device = y_mc.device
    T = len(sigma_schedule)

    # Per-coil scaling for σ_{c,i}^2 = σ_i^2 · (σ_c^2 / mean σ_c^2)
    sig_c = sense_op.sigma_c_sq.to(device)
    sig_c_norm = sig_c / sig_c.mean().clamp(min=eps)
    sig_c_view = sig_c_norm.view(Nc, 1, 1)

    sigma_i_sq = sigma_i_sq_init.clone().to(device)         # (H,W)

    # Mutable state — mask, sens, y can change with active acquisition / refine.
    mask = sense_op.mask.clone()
    sens = sense_op.sens.clone()
    inv_sumS2 = 1.0 / ((sens.abs() ** 2).sum(dim=0) + sense_op.eps)
    y_active = y_mc.clone()

    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)

    # Active acquisition schedule: which timesteps fire, and how many lines each.
    if active_lines > 0 and y_mc_full is not None and active_rounds >= 1:
        n_rounds = max(1, active_rounds)
        first = int(active_after_frac * T)
        last = int(active_until_frac * T)
        if n_rounds == 1:
            active_steps = [first]
        else:
            active_steps = [int(round(first + i * (last - first) / (n_rounds - 1)))
                            for i in range(n_rounds)]
        # Distribute lines as evenly as possible
        base = active_lines // n_rounds
        extra = active_lines - base * n_rounds
        active_per_round = {s: base + (1 if i < extra else 0)
                            for i, s in enumerate(active_steps)}
    else:
        active_per_round = {}

    psnr_traj = []
    diagnostics = {"P_maps": {}, "K_maps": {}, "sens_iters": [],
                   "active_cols": [], "num_lines_per_step": []}
    gen_probe = torch.Generator(device="cpu").manual_seed(seed + 991)

    active_done = False
    for step in range(T):
        sigma_t = sigma_schedule[step].item()
        mu = denoiser_fn(x_t, sigma_t)                  # (H,W)

        # ----- (A) posterior variance estimate -----
        if use_pv_gate:
            P, _ = estimate_posterior_variance_kspace(
                x_t=x_t, mu=mu, sigma_t=sigma_t,
                denoiser_fn=denoiser_fn,
                eps_probe=pv_eps_probe,
                centered=pv_centered,
                radial_smooth=pv_radial_smooth,
                radial_degree=pv_radial_degree,
                n_probes=pv_n_probes,
                generator=gen_probe,
            )                                            # (H,W) real positive
            if pv_normalize:
                # Rescale so that mean(P) = σ_t^2 — preserves the global
                # scale of the original FA-KGD gain while distributing
                # the σ_t^2 budget across frequencies. K then has the same
                # *average* magnitude but a learned per-i shape.
                P = P * (sigma_t ** 2 / P.mean().clamp(min=eps))
            # σ_{c,i}^2 broadcast across coils
            sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view     # (Nc,H,W)
            P_ci = P.unsqueeze(0).expand(Nc, -1, -1)             # (Nc,H,W)
            K = P_ci / (P_ci + sig_ci_sq)                         # (Nc,H,W)
        else:
            # Fall back to the original FA-KGD gain (no FPDC).
            sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view
            K = (sigma_t ** 2) / (sigma_t ** 2 + sig_ci_sq)
            P = None

        # ----- standard k-space residual -----
        mu_kc = fft2c(sens * mu.unsqueeze(0))            # (Nc,H,W)
        residual = mask * (y_active - mu_kc)

        # SENSE-combined correction with current sens / inv_sumS2
        coil_corr = ifft2c(K * residual)                 # (Nc,H,W)
        corr = (torch.conj(sens) * coil_corr).sum(dim=0) * inv_sumS2

        x_corrected = mu + corr

        # ----- (B) in-loop sens refinement -----
        if (refine_sens
            and step >= int(sens_warmup_frac * T)
            and (step % sens_refresh_every == 0)
            and step < T - 1):
            sens = refine_sens_maps_inloop(
                mc_y_obs=mask * y_active, mask=mask,
                x_hat=x_corrected, sens_prev=sens,
                cutoff_frac=sens_lp_cutoff, blend=sens_blend,
            )
            inv_sumS2 = 1.0 / ((sens.abs() ** 2).sum(dim=0) + sense_op.eps)
            if return_diagnostics:
                diagnostics["sens_iters"].append(step)

        # ----- (D) active line selection (multi-round) -----
        if active_per_round and step in active_per_round:
            n_lines_now = active_per_round[step]
            mask_col = (mask.sum(dim=0) > 0).float()
            avail_idx = torch.nonzero(mask_col < 0.5, as_tuple=False).flatten()
            n_avail = int(avail_idx.numel())
            k = min(n_lines_now, n_avail)
            new_cols: list = []
            if k > 0:
                if active_score == "pv" and P is not None:
                    score_per_col = (P / (sigma_i_sq + eps)).sum(dim=0)
                    score_per_col = score_per_col * (1.0 - mask_col)
                    new_cols = torch.topk(score_per_col, k=k).indices.tolist()
                elif active_score == "energy":
                    # Image-conditional: weight columns by predicted k-space
                    # energy of S * mu_theta(x_t). Measures "where the
                    # diffusion prior thinks the signal lives", independent
                    # of the (saturated) PV gate.
                    pred_k = fft2c(sens * mu.unsqueeze(0))           # (Nc,H,W)
                    score_per_col = (pred_k.abs() ** 2).sum(dim=(0, 1))  # (W,)
                    score_per_col = score_per_col * (1.0 - mask_col)
                    new_cols = torch.topk(score_per_col, k=k).indices.tolist()
                elif active_score == "residual_oracle":
                    # ORACLE upper bound: pick columns with the most
                    # *unexplained* k-space energy under current iterate,
                    # measured against the (in real MRI unavailable) full
                    # k-space. This is the best a residual-based scoring
                    # could ever do; not a deployable method.
                    if y_mc_full is not None:
                        pred_k = fft2c(sens * mu.unsqueeze(0))
                        unexp = (y_mc_full - pred_k).abs() ** 2     # (Nc,H,W)
                        score_per_col = unexp.sum(dim=(0, 1))        # (W,)
                    else:
                        score_per_col = (residual.abs() ** 2).sum(dim=(0, 1))
                    score_per_col = score_per_col * (1.0 - mask_col)
                    new_cols = torch.topk(score_per_col, k=k).indices.tolist()
                elif active_score == "random":
                    # Use a deterministic CPU generator so different seeds give
                    # reproducible random-adaptive masks.
                    g_act = torch.Generator(device="cpu").manual_seed(
                        seed + 7919 * (step + 1))
                    perm = torch.randperm(n_avail, generator=g_act)
                    new_cols = avail_idx[perm[:k]].tolist()
                elif active_score == "equispaced":
                    # Pick the columns that maximally fill the largest gaps in
                    # the current 1-D mask. Greedy: each pick goes to the
                    # midpoint of the longest unselected run.
                    sel = (mask_col > 0.5).cpu().numpy().astype(bool).copy()
                    chosen = []
                    import numpy as _np
                    for _ in range(k):
                        # find longest run of False
                        best_len, best_mid = -1, -1
                        i = 0
                        L = sel.size
                        while i < L:
                            if not sel[i]:
                                j = i
                                while j < L and not sel[j]:
                                    j += 1
                                run_len = j - i
                                if run_len > best_len:
                                    best_len = run_len
                                    best_mid = (i + j - 1) // 2
                                i = j
                            else:
                                i += 1
                        if best_mid < 0:
                            break
                        sel[best_mid] = True
                        chosen.append(best_mid)
                    new_cols = chosen
                else:
                    raise ValueError(f"unknown active_score={active_score!r}")
            if y_mc_full is not None and new_cols:
                if return_diagnostics:
                    diagnostics["active_cols"].extend(new_cols)
                for c in new_cols:
                    mask[:, c] = 1.0
                y_active = torch.where(mask > 0, y_mc_full, y_active)

        if return_diagnostics:
            diagnostics["num_lines_per_step"].append(int((mask.sum(dim=0) > 0).sum().item()))
            # Per-step divergence of PV gain from the scalar PiGDM gain — if
            # this stays ~0 the PV gate is collapsing (Issue #1 in the
            # post-mortem). Reference: K_pigdm = sigma_t^2/(sigma_t^2+sigma^2_ci)
            if P is not None:
                K_ref = (sigma_t ** 2) / (sigma_t ** 2 + sig_ci_sq)
                kdiv = (K - K_ref).abs().mean().item() / max(
                    K_ref.abs().mean().item(), 1e-12)
                diagnostics.setdefault("K_div_vs_pigdm", []).append(kdiv)
                # Angular structure of P: how much non-radial variance survives
                # the smoother. Computed as 1 − Var(radial_mean(P)) / Var(P).
                P_flat = P.detach()
                p_var = float(P_flat.var().clamp(min=1e-12).item())
                # cheap radial mean: fftshifted radius bucketing
                if "_rgrid" not in diagnostics:
                    fy = torch.arange(H, device=device).float() - H / 2
                    fx = torch.arange(W, device=device).float() - W / 2
                    diagnostics["_rgrid"] = (
                        torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
                        .round().long())
                rg = diagnostics["_rgrid"]
                P_shifted = torch.fft.fftshift(P_flat)
                rad_mean = torch.zeros(int(rg.max().item()) + 1,
                                        device=device)
                cnt = torch.zeros_like(rad_mean)
                rad_mean.scatter_add_(0, rg.flatten(),
                                      P_shifted.flatten())
                cnt.scatter_add_(0, rg.flatten(),
                                  torch.ones_like(P_shifted.flatten()))
                rad_mean = rad_mean / cnt.clamp(min=1)
                radial_pred = rad_mean[rg]
                ang_var = float((P_shifted - radial_pred).var().item())
                diagnostics.setdefault("P_ang_frac", []).append(
                    ang_var / max(p_var, 1e-12))
            if step in [0, T // 4, T // 2, 3 * T // 4, T - 1] and P is not None:
                diagnostics["P_maps"][step] = P.detach().cpu().numpy()
                diagnostics["K_maps"][step] = K.detach().mean(dim=0).cpu().numpy()

        if step < T - 1:
            sigma_next = sigma_schedule[step + 1].item()
            x_t = x_corrected + sigma_next * _complex_randn(H, W, device=device)
        else:
            x_t = x_corrected

        if x_gt is not None:
            psnr_traj.append(_psnr(x_gt, x_t))

    # Final SENSE combination uses the final (possibly refined) maps
    out = {"recon": x_t, "psnr_trajectory": psnr_traj,
           "sens_final": sens, "mask_final": mask}
    if return_diagnostics:
        diagnostics.pop("_rgrid", None)
        out["diagnostics"] = diagnostics
    return out
