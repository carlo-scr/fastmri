"""Legacy PVAS sampler plus corrected protocol utilities.

This module remains for backward compatibility while the operator-space
SketchSense sampler is developed. Its single-coil Fourier PV path is not a
valid multi-coil SENSE covariance model and must not support new multi-coil
paper claims. Correct candidate-operator sketches live in
``operator_sketch.py``.

Components on top of `run_fakgd_mc`:

  (A) Posterior-Variance-Gated DC.   The Kalman gain is

          K_{c,i}(t) = P_i(t) / (P_i(t) + σ_{c,i}^2)

      with P_i(t) the Tweedie second-moment estimate of the per-frequency
      posterior variance (see `posterior_var.py`). Replaces FPDC's
      hard-radius gate with a soft, theory-derived per-frequency gain.
      At low t (early sampling) P_i is small at high frequencies → small
      gain → automatic deferral, exactly mirroring FPDC. At high t (late
      sampling) P_i grows so all frequencies receive their measurement.

  (B) Experimental in-loop SENSE refinement. Every `sens_refresh_every` steps, refine
      the sensitivity maps using the current iterate's magnitude:

          S_c <- LowPass( IFFT(M*y_c) / x̂_0 )

      enforced smooth via a Gaussian low-pass in image domain. This is a
      diffusion analogue of JSENSE; no priority claim is made.

  (D) Active line selection.  After `active_warmup_frac * T` steps, query
      `num_active_lines` extra k-space columns ranked by the legacy marginal
      score

          score(col) = sum_r∈col P_r(t) / σ_{c,r}^2.

      We then "acquire" them by setting `mask[col]=1` and treating the
      ground-truth k-space at those lines as the new measurements (since
      this is a retrospective study). Optional via `active_*` args.
"""
from __future__ import annotations

import torch
from typing import Callable, Optional

from .mri_forward import fft2c, ifft2c, build_radius_grid
from .sense import MultiCoilSENSE
from .posterior_var import estimate_posterior_variance_kspace


def _complex_randn(H, W, device):
    """EDM noise with std σ per real network channel when multiplied by σ."""
    return torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)


def build_active_schedule(
    total_steps: int,
    total_atoms: int,
    rounds: int,
    after_frac: float,
    until_frac: float,
) -> dict[int, int]:
    """Return a unique ``step -> atoms`` acquisition schedule.

    Rounding a linspace into a set/dict can silently merge rounds and lose
    budget. This helper chooses unique integer steps and asserts that the
    requested budget is preserved.
    """
    if total_atoms <= 0:
        return {}
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    n_rounds = max(1, int(rounds))
    lo = min(total_steps - 1, max(0, int(round(after_frac * (total_steps - 1)))))
    hi = min(total_steps - 1, max(lo, int(round(until_frac * (total_steps - 1)))))
    candidates = list(range(lo, hi + 1))
    if n_rounds > len(candidates):
        raise ValueError(
            f"{n_rounds} acquisition rounds do not fit in unique step window "
            f"[{lo}, {hi}]"
        )
    if n_rounds == 1:
        steps = [lo]
    else:
        steps = [
            candidates[round(i * (len(candidates) - 1) / (n_rounds - 1))]
            for i in range(n_rounds)
        ]
    if len(set(steps)) != len(steps):
        raise AssertionError(f"non-unique acquisition steps: {steps}")
    base, extra = divmod(int(total_atoms), n_rounds)
    schedule = {step: base + (1 if i < extra else 0)
                for i, step in enumerate(steps)}
    if sum(schedule.values()) != int(total_atoms):
        raise AssertionError("active schedule lost measurement budget")
    return schedule


def random_complement_order(mask: torch.Tensor, seed: int) -> list[int]:
    """Deterministic random ordering of columns absent from ``mask``.

    The same ordering is shared by up-front and mid-trajectory random
    baselines so their final masks are identical and only timing differs.
    """
    mask_col = (mask.sum(dim=0) > 0) if mask.dim() == 2 else (mask > 0)
    available = torch.nonzero(~mask_col, as_tuple=False).flatten().cpu()
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    perm = torch.randperm(int(available.numel()), generator=gen)
    return available[perm].tolist()


def _topk_available(
    score_per_col: torch.Tensor,
    available: torch.Tensor,
    k: int,
) -> list[int]:
    """Top-k column indices restricted strictly to the available set."""
    if k <= 0 or available.numel() == 0:
        return []
    local = torch.topk(score_per_col[available], k=min(k, available.numel())).indices
    return available[local].tolist()


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
    active_score: str = "pv",             # legacy pv | random | equispaced
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
    initial_cols = int((mask.sum(dim=0) > 0).sum().item())
    sens = sense_op.sens.clone()
    inv_sumS2 = 1.0 / ((sens.abs() ** 2).sum(dim=0) + sense_op.eps)
    y_active = y_mc.clone()

    sigma_max = sigma_schedule[0].item()
    x_t = sigma_max * _complex_randn(H, W, device=device)

    # Active acquisition schedule: which timesteps fire, and how many lines each.
    active_per_round = (
        build_active_schedule(
            T, active_lines, active_rounds,
            active_after_frac, active_until_frac,
        )
        if active_lines > 0 and y_mc_full is not None
        else {}
    )
    random_order = random_complement_order(mask, seed) if active_score == "random" else []
    random_cursor = 0

    psnr_traj = []
    diagnostics = {"P_maps": {}, "K_maps": {}, "sens_iters": [],
                   "active_cols": [], "num_lines_per_step": []}
    gen_probe = torch.Generator(device="cpu").manual_seed(seed + 991)

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
                # Rescale so mean(P) is the total complex prior variance
                # 2σ_t², directly comparable with sigma_i_sq = E|n_i|².
                # This preserves the global gain scale while distributing
                # the prior budget across frequencies. K then has the same
                # *average* magnitude but a learned per-i shape.
                P = P * (2.0 * sigma_t ** 2 / P.mean().clamp(min=eps))
            # σ_{c,i}^2 broadcast across coils
            sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view     # (Nc,H,W)
            P_ci = P.unsqueeze(0).expand(Nc, -1, -1)             # (Nc,H,W)
            K = P_ci / (P_ci + sig_ci_sq)                         # (Nc,H,W)
        else:
            # Fall back to the original FA-KGD gain (no FPDC).
            sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view
            prior_var_complex = 2.0 * sigma_t ** 2
            K = prior_var_complex / (prior_var_complex + sig_ci_sq)
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
                if active_score == "pv" and P is None:
                    # Selection-only mode (gate off): compute the PV estimate
                    # on demand, only at acquisition rounds. Costs
                    # 2*pv_n_probes denoiser calls per round instead of per
                    # step; the DC gain stays the scalar FA-KGD/PiGDM one.
                    P, _ = estimate_posterior_variance_kspace(
                        x_t=x_t, mu=mu, sigma_t=sigma_t,
                        denoiser_fn=denoiser_fn,
                        eps_probe=pv_eps_probe,
                        centered=pv_centered,
                        radial_smooth=pv_radial_smooth,
                        radial_degree=pv_radial_degree,
                        n_probes=pv_n_probes,
                        generator=gen_probe,
                    )
                if active_score == "pv" and P is not None:
                    score_per_col = (P / (sigma_i_sq + eps)).sum(dim=0)
                    new_cols = _topk_available(score_per_col, avail_idx, k)
                elif active_score == "energy":
                    # Image-conditional: weight columns by predicted k-space
                    # energy of S * mu_theta(x_t). Measures "where the
                    # diffusion prior thinks the signal lives", independent
                    # of the (saturated) PV gate.
                    pred_k = fft2c(sens * mu.unsqueeze(0))           # (Nc,H,W)
                    score_per_col = (pred_k.abs() ** 2).sum(dim=(0, 1))  # (W,)
                    new_cols = _topk_available(score_per_col, avail_idx, k)
                elif active_score == "residual_oracle":
                    # Retrospective greedy-oracle reference: pick columns with
                    # the most unexplained energy under the current iterate,
                    # measured against unavailable full k-space. This is not a
                    # bound on final reconstruction utility.
                    if y_mc_full is not None:
                        pred_k = fft2c(sens * mu.unsqueeze(0))
                        unexp = (y_mc_full - pred_k).abs() ** 2     # (Nc,H,W)
                        score_per_col = unexp.sum(dim=(0, 1))        # (W,)
                    else:
                        score_per_col = (residual.abs() ** 2).sum(dim=(0, 1))
                    new_cols = _topk_available(score_per_col, avail_idx, k)
                elif active_score == "random":
                    # Reveal the next entries of one fixed random complement
                    # ranking. Static and adaptive random controls share this
                    # exact final mask.
                    chosen = []
                    while len(chosen) < k and random_cursor < len(random_order):
                        c = random_order[random_cursor]
                        random_cursor += 1
                        if mask_col[c] < 0.5:
                            chosen.append(c)
                    new_cols = chosen
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
                # Apply newly acquired data immediately, matching the ensemble
                # policy and the natural-image implementation. Otherwise the
                # final acquisition round receives one fewer correction.
                residual = mask * (y_active - mu_kc)
                coil_corr = ifft2c(K * residual)
                corr = (torch.conj(sens) * coil_corr).sum(dim=0) * inv_sumS2
                x_corrected = mu + corr

        if return_diagnostics:
            diagnostics["num_lines_per_step"].append(int((mask.sum(dim=0) > 0).sum().item()))
            # Per-step divergence of PV gain from the scalar PiGDM gain — if
            # this stays ~0 the PV gate is collapsing (Issue #1 in the
            # post-mortem). The reference uses total complex prior variance
            # 2σ_t² and complex measurement variance E|n|².
            if P is not None:
                prior_var_complex = 2.0 * sigma_t ** 2
                K_ref = prior_var_complex / (prior_var_complex + sig_ci_sq)
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
    if active_per_round:
        expected = initial_cols + min(active_lines, W - initial_cols)
        actual = int((mask.sum(dim=0) > 0).sum().item())
        if actual != expected:
            raise AssertionError(
                f"active budget mismatch: expected {expected} columns, got {actual}"
            )
    if return_diagnostics:
        diagnostics.pop("_rgrid", None)
        out["diagnostics"] = diagnostics
    return out


def run_ensemble_active_mc(
    y_mc: torch.Tensor,
    sense_op: MultiCoilSENSE,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_i_sq_init: torch.Tensor,
    n_particles: int = 4,
    eps: float = 1e-8,
    active_lines: int = 0,
    active_rounds: int = 1,
    active_after_frac: float = 0.5,
    active_until_frac: float = 0.85,
    active_score: str = "variance",       # "variance" | "random"
    y_mc_full: Optional[torch.Tensor] = None,
    seed: int = 0,
) -> dict:
    """ADS/AdaSense-style ensemble reference for multi-coil MRI.

    N parallel PiGDM particle trajectories (scalar gain, no PV gate) share the
    measurement state; at each acquisition round, unscanned k-space columns are
    scored by the CROSS-PARTICLE variance of the particles' predicted coil
    k-space (Tweedie means pushed through the forward model) --- the
    measurement-space disagreement signal that ADS (Nolan et al., TMLR 2025)
    and AdaSense (Elata et al., ECCV 2024) obtain from posterior-sample
    ensembles. Reconstruction = complex mean of the particle recons.
    NFE = n_particles * T. `active_score="random"` isolates the pure
    particle-averaging benefit (selection value = variance minus random).
    """
    torch.manual_seed(seed)
    Nc, H, W = y_mc.shape
    device = y_mc.device
    T = len(sigma_schedule)
    N = n_particles

    sig_c = sense_op.sigma_c_sq.to(device)
    sig_c_norm = sig_c / sig_c.mean().clamp(min=eps)
    sig_c_view = sig_c_norm.view(Nc, 1, 1)
    sigma_i_sq = sigma_i_sq_init.clone().to(device)

    mask = sense_op.mask.clone()
    initial_cols = int((mask.sum(dim=0) > 0).sum().item())
    sens = sense_op.sens.clone()
    inv_sumS2 = 1.0 / ((sens.abs() ** 2).sum(dim=0) + sense_op.eps)
    y_active = y_mc.clone()

    sigma_max = sigma_schedule[0].item()
    xs = [sigma_max * _complex_randn(H, W, device=device) for _ in range(N)]

    active_per_round = (
        build_active_schedule(
            T, active_lines, active_rounds,
            active_after_frac, active_until_frac,
        )
        if active_lines > 0 and y_mc_full is not None
        else {}
    )
    random_order = random_complement_order(mask, seed) if active_score == "random" else []
    random_cursor = 0

    nfe = 0
    for step in range(T):
        sigma_t = sigma_schedule[step].item()
        mus, preds = [], []
        for n in range(N):
            mu = denoiser_fn(xs[n], sigma_t)
            nfe += 1
            mus.append(mu)
            preds.append(fft2c(sens * mu.unsqueeze(0)))       # (Nc,H,W)

        # ----- acquisition round: cross-particle variance in k-space -----
        if active_per_round and step in active_per_round:
            n_lines_now = active_per_round[step]
            mask_col = (mask.sum(dim=0) > 0).float()
            avail_idx = torch.nonzero(mask_col < 0.5, as_tuple=False).flatten()
            k = min(n_lines_now, int(avail_idx.numel()))
            new_cols: list = []
            if k > 0:
                if active_score == "variance":
                    stack = torch.stack(preds, dim=0)          # (N,Nc,H,W)
                    var = stack.var(dim=0).sum(dim=(0, 1)).real  # (W,)
                    new_cols = _topk_available(var, avail_idx, k)
                elif active_score == "random":
                    chosen = []
                    while len(chosen) < k and random_cursor < len(random_order):
                        c = random_order[random_cursor]
                        random_cursor += 1
                        if mask_col[c] < 0.5:
                            chosen.append(c)
                    new_cols = chosen
                else:
                    raise ValueError(active_score)
            if y_mc_full is not None and new_cols:
                for c in new_cols:
                    mask[:, c] = 1.0
                y_active = torch.where(mask > 0, y_mc_full, y_active)

        # ----- scalar-gain PiGDM correction per particle -----
        sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view
        prior_var_complex = 2.0 * sigma_t ** 2
        K = prior_var_complex / (prior_var_complex + sig_ci_sq)
        for n in range(N):
            residual = mask * (y_active - preds[n])
            corr = (torch.conj(sens) * ifft2c(K * residual)).sum(dim=0) * inv_sumS2
            xc = mus[n] + corr
            if step < T - 1:
                sigma_next = sigma_schedule[step + 1].item()
                xs[n] = xc + sigma_next * _complex_randn(H, W, device=device)
            else:
                xs[n] = xc

    recon = torch.stack(xs, dim=0).mean(dim=0)
    if active_per_round:
        expected = initial_cols + min(active_lines, W - initial_cols)
        actual = int((mask.sum(dim=0) > 0).sum().item())
        if actual != expected:
            raise AssertionError(
                f"active budget mismatch: expected {expected} columns, got {actual}"
            )
    return {"recon": recon, "mask_final": mask, "nfe": nfe,
            "particle_recons": xs}
