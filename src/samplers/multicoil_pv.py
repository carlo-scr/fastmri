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
from .operator_sketch import (
    build_atom_information,
    conditional_atom_scores,
    diagonal_atom_scores,
    expand_measurement_values_to_real,
    greedy_logdet_select,
    real_unflatten,
    sketch_operator_posterior_covariance,
)
from .rao_belief import (
    RaoBlackwellizedBelief,
    branch_from_operator_sketch,
    plan_horizon_two,
    screen_actions,
)
from .sense import MultiCoilSENSE
from .posterior_var import estimate_posterior_variance_kspace


def _complex_randn(H, W, device):
    """EDM noise with std σ per real network channel when multiplied by σ."""
    return torch.randn(H, W, device=device) + 1j * torch.randn(H, W, device=device)


def _probe_generator(seed: int, step: int) -> torch.Generator:
    """Counter-based probe stream shared by gate-on/off ablations."""
    probe_seed = int(
        (int(seed) * 1_000_003 + int(step) * 104_729 + 17)
        % (2**63 - 1)
    )
    return torch.Generator(device="cpu").manual_seed(probe_seed)


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
    sigma_i_sq_init: torch.Tensor,       # total complex variance E|n_i|^2
    eps: float = 1e-8,
    # (A) posterior-variance gating
    use_pv_gate: bool = True,
    pv_eps_probe: float = 5e-2,
    pv_radial_smooth: bool = True,
    pv_radial_degree: int = 4,
    pv_centered: bool = True,
    pv_normalize: bool = True,            # rescale mean(P̂) to 2σ_t²
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
    active_score: str = "pv",             # legacy pv | operator/sketch policies
    sketch_rank: int = 4,
    sketch_fd_step: float = 5e-2,
    sketch_psd_rtol: float = 1e-6,
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
                   "active_cols": [], "num_lines_per_step": [],
                   "operator_sketches": []}
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
                generator=_probe_generator(seed, step),
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
            # An acquisition can fire on the same step. Its immediate
            # correction must use the refreshed forward operator as well as
            # the refreshed adjoint/normalization.
            mu_kc = fft2c(sens * mu.unsqueeze(0))
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
                if active_score in {
                    "operator_diag", "sketch_marginal", "sketch_logdet"
                }:
                    # Full unmasked SENSE candidate bank B_c x = F(S_c x).
                    # The pullback is the true Hilbert adjoint, not the
                    # normalized/masked MultiCoilSENSE.adjoint pseudoinverse.
                    def candidate_forward(value):
                        return fft2c(sens * value.unsqueeze(0))

                    def candidate_adjoint(value):
                        return (
                            torch.conj(sens) * ifft2c(value)
                        ).sum(dim=0)

                    measurement_template = candidate_forward(mu)
                    sketch = sketch_operator_posterior_covariance(
                        x_t=x_t,
                        mu=mu,
                        sigma_t=sigma_t,
                        denoiser=denoiser_fn,
                        forward=candidate_forward,
                        adjoint=candidate_adjoint,
                        measurement_template=measurement_template,
                        num_probes=sketch_rank,
                        fd_step=sketch_fd_step,
                        centered=pv_centered,
                        generator=_probe_generator(seed + 8_000_003, step),
                        psd_rtol=sketch_psd_rtol,
                    )
                    atom_ids = torch.arange(
                        W, device=device, dtype=torch.long
                    ).view(1, 1, W).expand(Nc, H, W)
                    if active_score == "operator_diag":
                        score_per_col = diagonal_atom_scores(
                            sketch,
                            atom_ids,
                            sig_ci_sq,
                            num_atoms=W,
                            complex_noise_is_total=True,
                            source="hutchinson",
                        )
                        new_cols = _topk_available(
                            score_per_col, avail_idx, k
                        )
                        selected_gains = [
                            float(score_per_col[c].item()) for c in new_cols
                        ]
                    else:
                        atom_information = build_atom_information(
                            sketch,
                            atom_ids,
                            sig_ci_sq,
                            num_atoms=W,
                            complex_noise_is_total=True,
                        )
                        if active_score == "sketch_marginal":
                            observed_idx = torch.nonzero(
                                mask_col > 0.5, as_tuple=False
                            ).flatten()
                            conditioned_scores = conditional_atom_scores(
                                atom_information,
                                conditioned_on=observed_idx,
                            )
                            new_cols = _topk_available(
                                conditioned_scores,
                                avail_idx,
                                k,
                            )
                            selected_gains = [
                                float(
                                    conditioned_scores[c].item()
                                )
                                for c in new_cols
                            ]
                        else:
                            observed_idx = torch.nonzero(
                                mask_col > 0.5, as_tuple=False
                            ).flatten()
                            new_cols, selected_gains = greedy_logdet_select(
                                atom_information,
                                k,
                                eligible=avail_idx,
                                conditioned_on=observed_idx,
                            )
                    if return_diagnostics:
                        diagnostics["operator_sketches"].append({
                            "step": int(step),
                            "sigma_t": float(sigma_t),
                            "policy": active_score,
                            "selected_cols": [int(c) for c in new_cols],
                            "selected_gains": [
                                float(g) for g in selected_gains
                            ],
                            **sketch.diagnostics,
                        })
                elif active_score == "pv":
                    if P is None:
                        # Selection-only mode (gate off): compute the legacy
                        # Fourier score on demand at acquisition rounds.
                        P, _ = estimate_posterior_variance_kspace(
                            x_t=x_t, mu=mu, sigma_t=sigma_t,
                            denoiser_fn=denoiser_fn,
                            eps_probe=pv_eps_probe,
                            centered=pv_centered,
                            radial_smooth=pv_radial_smooth,
                            radial_degree=pv_radial_degree,
                            n_probes=pv_n_probes,
                            generator=_probe_generator(seed, step),
                        )
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
    """Shared-pipeline particle-covariance reference for multi-coil MRI.

    N parallel PiGDM particle trajectories (scalar gain, no PV gate) share the
    measurement state; at each acquisition round, unscanned k-space columns are
    scored by the CROSS-PARTICLE variance of the particles' predicted coil
    k-space (Tweedie means pushed through the forward model). This isolates a
    measurement-space disagreement mechanism inspired by ADS and AdaSense; it
    is not a faithful reproduction of either published algorithm.
    Reconstruction = complex mean of the particle recons.
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


def run_raosense_active_mc(
    y_mc: torch.Tensor,
    sense_op: MultiCoilSENSE,
    sigma_schedule: torch.Tensor,
    denoiser_fn: Callable[[torch.Tensor, float], torch.Tensor],
    sigma_i_sq_init: torch.Tensor,
    *,
    n_branches: int = 8,
    tangent_rank: int = 2,
    active_lines: int = 0,
    active_rounds: int = 4,
    active_after_frac: float = 0.35,
    active_until_frac: float = 0.80,
    planning_horizon: int = 2,
    beam_width: int = 8,
    num_fantasies: int = 16,
    information_samples: int = 128,
    planner_candidate_cap: int | None = 48,
    use_belief_correction: bool = True,
    sketch_fd_step: float = 5e-2,
    sketch_psd_rtol: float = 1e-6,
    y_mc_full: Optional[torch.Tensor] = None,
    seed: int = 0,
    return_diagnostics: bool = False,
    eps: float = 1e-8,
) -> dict:
    """RaoSense active multi-coil MRI with analytic outcome planning.

    ``n_branches`` measurement-guided trajectories represent between-mode
    uncertainty.  At each acquisition checkpoint, a rank-``tangent_rank``
    operator-space sketch supplies within-branch uncertainty.  The resulting
    Gaussian mixture is conditioned and planned entirely in its sufficient
    subspace.  Fantasy lookahead adds no denoiser calls.

    The denoiser-equivalent count is

    ``n_branches * T + 2 * n_branches * tangent_rank * active_rounds``

    when every requested round fires.  The implementation is a retrospective
    Cartesian-line simulator: ``y_mc_full`` is accessed only after a line is
    selected.  It is not a scanner-control implementation.
    """
    if n_branches <= 0:
        raise ValueError("n_branches must be positive")
    if tangent_rank <= 0:
        raise ValueError("tangent_rank must be positive")
    if planning_horizon not in {1, 2}:
        raise ValueError("planning_horizon must be one or two")
    if planner_candidate_cap is not None and planner_candidate_cap <= 0:
        raise ValueError("planner_candidate_cap must be positive or None")

    torch.manual_seed(seed)
    Nc, H, W = y_mc.shape
    device = y_mc.device
    T = len(sigma_schedule)

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
    xs = [
        sigma_max * _complex_randn(H, W, device=device)
        for _ in range(n_branches)
    ]
    branch_weights = torch.full(
        (n_branches,),
        1.0 / n_branches,
        device=device,
        dtype=y_mc.real.dtype,
    )

    active_per_round = (
        build_active_schedule(
            T,
            active_lines,
            active_rounds,
            active_after_frac,
            active_until_frac,
        )
        if active_lines > 0 and y_mc_full is not None
        else {}
    )

    # Each Cartesian action contains every coil/readout sample at one
    # phase-encode column, in stacked real/imaginary flattening order.
    complex_dimension = Nc * H * W
    row_bases = torch.arange(
        Nc * H, device=device, dtype=torch.long
    ) * W
    line_indices = {
        column: torch.cat(
            (
                row_bases + column,
                row_bases + column + complex_dimension,
            )
        )
        for column in range(W)
    }

    nfe = 0
    diagnostics = {
        "acquisitions": [],
        "weight_trajectory": [],
        "effective_sample_size": [],
        "nfe_base": 0,
        "nfe_tangent": 0,
    }

    for step in range(T):
        sigma_t = sigma_schedule[step].item()
        mus = []
        predictions = []
        for branch_index in range(n_branches):
            mu = denoiser_fn(xs[branch_index], sigma_t)
            nfe += 1
            diagnostics["nfe_base"] += 1
            mus.append(mu)
            predictions.append(fft2c(sens * mu.unsqueeze(0)))

        if active_per_round and step in active_per_round:
            def candidate_forward(value):
                return fft2c(sens * value.unsqueeze(0))

            def candidate_adjoint(value):
                return (
                    torch.conj(sens) * ifft2c(value)
                ).sum(dim=0)

            belief_branches = []
            for branch_index in range(n_branches):
                sketch = sketch_operator_posterior_covariance(
                    x_t=xs[branch_index],
                    mu=mus[branch_index],
                    sigma_t=sigma_t,
                    denoiser=denoiser_fn,
                    forward=candidate_forward,
                    adjoint=candidate_adjoint,
                    measurement_template=predictions[branch_index],
                    num_probes=tangent_rank,
                    fd_step=sketch_fd_step,
                    centered=True,
                    generator=_probe_generator(
                        seed + 19_000_003 * (branch_index + 1),
                        step,
                    ),
                    psd_rtol=sketch_psd_rtol,
                )
                nfe += 2 * tangent_rank
                diagnostics["nfe_tangent"] += 2 * tangent_rank
                belief_branches.append(
                    branch_from_operator_sketch(
                        weight=float(branch_weights[branch_index].item()),
                        measurement_mean=predictions[branch_index],
                        state_mean=mus[branch_index],
                        sketch=sketch,
                        label=f"branch-{branch_index}",
                    )
                )
            belief = RaoBlackwellizedBelief(belief_branches)
            measurement_noise_total = (
                sigma_i_sq.unsqueeze(0) * sig_c_view
            )
            measurement_noise_real = expand_measurement_values_to_real(
                measurement_noise_total,
                predictions[0],
                complex_values_are_total=True,
            )
            full_observation_real = (
                torch.cat(
                    (
                        y_mc_full.real.reshape(-1),
                        y_mc_full.imag.reshape(-1),
                    )
                )
                if y_mc_full is not None
                else None
            )

            requested = active_per_round[step]
            for within_round in range(requested):
                mask_col = mask.sum(dim=0) > 0
                available = torch.nonzero(
                    ~mask_col, as_tuple=False
                ).flatten().tolist()
                if not available:
                    break
                actions = {
                    int(column): line_indices[int(column)]
                    for column in available
                }

                # Optional declared engineering cap.  Bounds are evaluated for
                # every feasible line; only the highest immediate-information
                # upper bounds enter Monte Carlo/lookahead.  This truncation
                # is not safe for H=2 because future value can reverse the
                # immediate ordering.  Set None for uncapped root search.
                cap_pruned = 0
                if (
                    planner_candidate_cap is not None
                    and len(actions) > planner_candidate_cap
                ):
                    upper_ranked = sorted(
                        actions,
                        key=lambda action: (
                            belief.information_bounds(
                                actions[action], measurement_noise_real
                            ).upper,
                            -action,
                        ),
                        reverse=True,
                    )
                    keep = set(upper_ranked[:planner_candidate_cap])
                    cap_pruned = len(actions) - len(keep)
                    actions = {
                        action: indices
                        for action, indices in actions.items()
                        if action in keep
                    }

                planning_generator = _probe_generator(
                    seed + 31_000_031 + within_round,
                    step,
                )
                if planning_horizon == 2 and len(actions) > 1:
                    plan = plan_horizon_two(
                        belief,
                        actions,
                        measurement_noise_real,
                        beam_width=min(beam_width, len(actions)),
                        num_fantasies=num_fantasies,
                        information_samples=information_samples,
                        generator=planning_generator,
                    )
                    selected = int(plan.action)
                    plan_diagnostics = {
                        "objective": float(plan.objective),
                        "immediate_information": float(
                            plan.immediate_information
                        ),
                        "expected_future_information": float(
                            plan.expected_future_information
                        ),
                        **dict(plan.diagnostics),
                    }
                else:
                    survivors, rows = screen_actions(
                        belief, actions, measurement_noise_real
                    )
                    estimates = {
                        action: belief.estimate_information(
                            actions[action],
                            measurement_noise_real,
                            num_samples=information_samples,
                            generator=planning_generator,
                        )
                        for action in survivors
                    }
                    selected = max(
                        survivors,
                        key=lambda action: (
                            estimates[action].total,
                            -action,
                        ),
                    )
                    plan_diagnostics = {
                        "objective": float(estimates[selected].total),
                        "immediate_information": float(
                            estimates[selected].total
                        ),
                        "expected_future_information": 0.0,
                        "root_survivors": len(survivors),
                        "safely_pruned_nodes": sum(
                            row.safely_pruned for row in rows
                        ),
                    }

                selected_indices = line_indices[selected]
                assert full_observation_real is not None
                observation = full_observation_real[selected_indices]
                belief = belief.condition(
                    selected_indices,
                    observation,
                    measurement_noise_real,
                )
                branch_weights = belief.weights
                if use_belief_correction:
                    for branch_index, branch in enumerate(belief.branches):
                        assert branch.state_mean is not None
                        mus[branch_index] = real_unflatten(
                            branch.state_mean,
                            mus[branch_index],
                        )
                        predictions[branch_index] = candidate_forward(
                            mus[branch_index]
                        )

                mask[:, selected] = 1.0
                y_active = torch.where(mask > 0, y_mc_full, y_active)
                weights_cpu = [
                    float(value) for value in branch_weights.detach().cpu()
                ]
                effective_sample_size = float(
                    branch_weights.square().sum().reciprocal().item()
                )
                if return_diagnostics:
                    within_trace = sum(
                        branch_weights[index]
                        * belief.branches[index]
                        .measurement_factor.square().sum()
                        for index in range(n_branches)
                    )
                    _, total_factor = belief.measurement_moments()
                    total_trace = total_factor.square().sum()
                    between_trace = (total_trace - within_trace).clamp_min(0)
                    diagnostics["acquisitions"].append({
                        "step": int(step),
                        "within_round": int(within_round),
                        "sigma_t": float(sigma_t),
                        "selected_column": int(selected),
                        "candidate_cap_pruned": int(cap_pruned),
                        "weights": weights_cpu,
                        "effective_sample_size": effective_sample_size,
                        "within_mode_trace": float(within_trace.item()),
                        "between_mode_trace": float(between_trace.item()),
                        "between_mode_fraction": float(
                            (
                                between_trace
                                / total_trace.clamp_min(
                                    torch.finfo(total_trace.dtype).eps
                                )
                            ).item()
                        ),
                        **plan_diagnostics,
                    })

        sig_ci_sq = sigma_i_sq.unsqueeze(0) * sig_c_view
        prior_var_complex = 2.0 * sigma_t ** 2
        gain = prior_var_complex / (prior_var_complex + sig_ci_sq)
        for branch_index in range(n_branches):
            prediction = fft2c(sens * mus[branch_index].unsqueeze(0))
            residual = mask * (y_active - prediction)
            correction = (
                torch.conj(sens) * ifft2c(gain * residual)
            ).sum(dim=0) * inv_sumS2
            corrected = mus[branch_index] + correction
            if step < T - 1:
                sigma_next = sigma_schedule[step + 1].item()
                xs[branch_index] = (
                    corrected
                    + sigma_next * _complex_randn(H, W, device=device)
                )
            else:
                xs[branch_index] = corrected

        if return_diagnostics:
            diagnostics["weight_trajectory"].append(
                [float(value) for value in branch_weights.detach().cpu()]
            )
            diagnostics["effective_sample_size"].append(
                float(branch_weights.square().sum().reciprocal().item())
            )

    stacked = torch.stack(xs, dim=0)
    weight_view = branch_weights.to(stacked.real.dtype).view(
        n_branches, *([1] * (stacked.dim() - 1))
    )
    recon = (weight_view * stacked).sum(dim=0)
    if active_per_round:
        expected = initial_cols + min(active_lines, W - initial_cols)
        actual = int((mask.sum(dim=0) > 0).sum().item())
        if actual != expected:
            raise AssertionError(
                f"active budget mismatch: expected {expected} columns, got {actual}"
            )
    output = {
        "recon": recon,
        "mask_final": mask,
        "nfe": nfe,
        "branch_recons": xs,
        "branch_weights": branch_weights,
    }
    if return_diagnostics:
        output["diagnostics"] = diagnostics
    return output
