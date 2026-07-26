"""Domain-general PVAS: posterior-variance-guided active sensing on a pretrained DDPM.

Implements, for pixel-domain measurement operators on natural images, the same
machinery as src/samplers/{posterior_var,multicoil_pv}.py implements for MRI:

  1. Tweedie second-moment posterior-variance estimation from a SINGLE sampling
     trajectory (Hutchinson probe + central finite-difference JVP),
  2. variance-gated (Kalman-style) data consistency, and
  3. mid-trajectory active measurement selection under a matched budget.

Prior: any diffusers UNet2DModel with epsilon prediction (VP/DDPM schedule).
Operators: block-pixel acquisition (active inpainting / "where to look next").

Notation (VP diffusion): x_t = sqrt(abar_t) x0 + sqrt(1-abar_t) eps.
Define xtilde = x_t / sqrt(abar_t) = x0 + sigma_t eps with
sigma_t^2 = (1-abar_t)/abar_t.  Tweedie: E[x0|x_t] = xtilde + sigma_t^2 * score.
Differentiating (Efron 2011):  Cov[x0|x_t] = sigma_t^2 * d mu / d xtilde.
We estimate diag(Cov) with a Hutchinson probe v ~ N(0, I):
  diag_est = sigma_t^2 * (J v) * v   (unbiased before finite-probe
  post-processing). The legacy positive score sigma_t^2 * (J v)^2 is retained
  only as a row-energy diagnostic: it estimates diag(J J^T), not diag(J).
  Experiments use (J v)*v averaged over probes and channels before one
  positivity projection and block pooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


# ---------------------------------------------------------------------------
# Schedule helpers (VP / DDPM, epsilon-prediction)
# ---------------------------------------------------------------------------

def make_abar(scheduler) -> torch.Tensor:
    """Cumulative alphas from a diffusers DDPMScheduler."""
    return scheduler.alphas_cumprod.clone()


def ddim_timesteps(num_train_steps: int, num_steps: int) -> torch.Tensor:
    """Evenly spaced DDIM timestep subsequence, descending."""
    ts = torch.linspace(num_train_steps - 1, 0, num_steps).round().long()
    return ts


@dataclass
class DenoiserStats:
    """Bookkeeping for NFE accounting."""
    nfe: int = 0


class EpsDenoiser:
    """Wraps a diffusers UNet2DModel into x0-prediction with NFE counting."""

    def __init__(self, unet, abar: torch.Tensor, stats: DenoiserStats | None = None):
        self.unet = unet
        self.abar = abar
        self.stats = stats if stats is not None else DenoiserStats()

    @torch.no_grad()
    def predict(self, xt: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(E[x0 | x_t], eps_theta) via epsilon prediction."""
        b = xt.shape[0]
        tt = torch.full((b,), int(t), device=xt.device, dtype=torch.long)
        eps = self.unet(xt, tt).sample
        self.stats.nfe += b
        a = self.abar[t].to(xt.device)
        return (xt - torch.sqrt(1.0 - a) * eps) / torch.sqrt(a), eps

    @torch.no_grad()
    def x0hat(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        """E[x0 | x_t] via epsilon prediction."""
        return self.predict(xt, t)[0]


# ---------------------------------------------------------------------------
# Measurement operator: block-pixel acquisition (active inpainting)
# ---------------------------------------------------------------------------

@dataclass
class BlockMaskState:
    """Observation state for k x k block-pixel acquisition.

    mask: (B,1,H,W) float in {0,1}; y: (B,C,H,W) noisy observed image
    (defined on mask support only); sigma_y: measurement noise std.
    y_lr: optional (B,C,H/b,W/b) box-averaged low-resolution observation
    (adaptive super-resolution mode: LR is always measured, the budget buys
    full-resolution blocks).
    y_full: fixed noisy full-resolution measurement field used by the
    retrospective protocol. Every policy/particle that acquires the same atom
    reads the same value rather than receiving a fresh noise realization.
    """
    mask: torch.Tensor
    y: torch.Tensor
    sigma_y: float
    block: int
    y_lr: torch.Tensor | None = None
    y_full: torch.Tensor | None = None

    def clone(self) -> "BlockMaskState":
        return BlockMaskState(self.mask.clone(), self.y.clone(), self.sigma_y,
                              self.block,
                              None if self.y_lr is None else self.y_lr.clone(),
                              None if self.y_full is None else self.y_full.clone())

    @property
    def n_blocks_observed(self) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(self.mask, self.block)
        return (pooled > 0.5).flatten(1).sum(dim=1)


def init_block_state(x0: torch.Tensor, frac: float, block: int, sigma_y: float,
                     gen: torch.Generator, with_lr: bool = False) -> BlockMaskState:
    """Observe a random `frac` of k x k blocks of the ground-truth image x0.

    with_lr=True additionally observes the box-averaged low-resolution image
    (adaptive super-resolution mode).
    """
    B, C, H, W = x0.shape
    gh, gw = H // block, W // block
    n_total = gh * gw
    n_obs = max(0, int(round(frac * n_total)))
    mask_small = torch.zeros(B, 1, gh, gw, device=x0.device)
    for b in range(B):
        if n_obs > 0:
            idx = torch.randperm(n_total, generator=gen)[:n_obs]
            mask_small[b, 0].view(-1)[idx.to(x0.device)] = 1.0
    mask = torch.nn.functional.interpolate(mask_small, scale_factor=block, mode="nearest")
    noise = torch.randn(x0.shape, generator=gen).to(x0.device)
    y_full = x0 + sigma_y * noise
    y = y_full * mask
    y_lr = torch.nn.functional.avg_pool2d(x0, block) if with_lr else None
    return BlockMaskState(
        mask=mask, y=y, sigma_y=sigma_y, block=block,
        y_lr=y_lr, y_full=y_full,
    )


def acquire_blocks(state: BlockMaskState, x0: torch.Tensor, scores: torch.Tensor,
                   k: int, gen: torch.Generator) -> BlockMaskState:
    """Acquire the top-k UNOBSERVED blocks by `scores` (B,1,gh,gw); returns new state.

    Retrospective protocol: new measurements are read from one fixed noisy
    measurement field generated when the state is initialized. Thus policies
    and particles that acquire the same atom receive the same value.
    """
    B = state.mask.shape[0]
    block = state.block
    pooled_mask = torch.nn.functional.avg_pool2d(state.mask, block)  # (B,1,gh,gw)
    scores = scores.masked_fill(pooled_mask > 0.5, float("-inf"))
    flat = scores.flatten(1)
    new_mask_small = torch.zeros_like(flat)
    take = min(k, flat.shape[1])
    top = flat.topk(take, dim=1).indices
    new_mask_small.scatter_(1, top, 1.0)
    # Guard against -inf rows (all blocks observed): only keep finite picks
    picked_valid = torch.gather(flat, 1, top).isfinite().float()
    new_mask_small.scatter_(1, top, picked_valid)
    gh = state.mask.shape[2] // block
    gw = state.mask.shape[3] // block
    new_mask = torch.nn.functional.interpolate(
        new_mask_small.view(B, 1, gh, gw), scale_factor=block, mode="nearest")
    if state.y_full is None:
        # Backward-compatible fallback for manually constructed states.
        noise = torch.randn(x0.shape, generator=gen).to(x0.device)
        y_full = x0 + state.sigma_y * noise
    else:
        y_full = state.y_full
    y_new = y_full * new_mask
    mask = torch.clamp(state.mask + new_mask, 0, 1)
    y = state.y + y_new * (1 - (state.mask > 0.5).float())  # don't double-write
    return BlockMaskState(mask=mask, y=y, sigma_y=state.sigma_y, block=block,
                          y_lr=state.y_lr, y_full=y_full)


# ---------------------------------------------------------------------------
# Posterior-variance estimator (Tweedie second moment, Hutchinson probe)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pv_estimate(den: EpsDenoiser, xt: torch.Tensor, t: int,
                eps_probe: float = 5e-2, n_probes: int = 1,
                pool_block: int | None = None,
                smooth_sigma: float | None = 1.0,
                mode: str = "jv_v",
                generator: torch.Generator | None = None,
                clamp_final: bool = True) -> torch.Tensor:
    """Estimate diag Cov[x0 | x_t] per pixel (averaged over channels).

    mode "jv_v":  sigma_t^2 * (Jv) * v  -- unbiased for the diagonal.
    mode "jv_sq": sigma_t^2 * (Jv)^2    -- positive squared-row-energy
        diagnostic; not a covariance diagonal.

    The probe is applied in the xtilde = x_t / sqrt(abar) domain, i.e. we
    perturb x_t by eps * sqrt(abar_t) * v so that the finite difference
    approximates (d mu / d xtilde) v.  Central differences cancel the O(eps)
    Taylor term, leaving O(eps^2) truncation error.  Returns (B,1,H,W) after
    channel-averaging, optional Gaussian smoothing, optional block sum-pooling,
    and (by default) one final floor at zero (B,1,gh,gw). Set
    ``clamp_final=False`` when a caller will aggregate signed coordinates into
    measurement atoms before applying the floor. Only the signed estimate is
    unbiased.
    """
    a = den.abar[t].to(xt.device)
    sig2 = ((1.0 - a) / a).item()
    scale = eps_probe * math.sqrt(a.item())
    est = torch.zeros_like(xt)
    for _ in range(n_probes):
        if generator is None:
            v = torch.randn_like(xt)
        else:
            v = torch.randn(xt.shape, generator=generator).to(
                device=xt.device, dtype=xt.dtype
            )
        mu_p = den.x0hat(xt + scale * v, t)
        mu_m = den.x0hat(xt - scale * v, t)
        jv = (mu_p - mu_m) / (2.0 * eps_probe)
        est = est + (jv * v if mode == "jv_v" else jv.pow(2))
    est = sig2 * est / n_probes
    # Average probes/channels before positivity projection. Clipping each
    # noisy channel separately creates a much larger positive bias.
    est = est.mean(dim=1, keepdim=True)
    if smooth_sigma is not None and smooth_sigma > 0:
        est = _gaussian_blur(est, smooth_sigma)
    if pool_block is not None:
        est = torch.nn.functional.avg_pool2d(est, pool_block) * (pool_block ** 2)
    return est.clamp_min(0.0) if clamp_final else est


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    # Reflect padding avoids systematically suppressing border uncertainty.
    x = torch.nn.functional.pad(x, (r, r, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.unsqueeze(2))
    x = torch.nn.functional.pad(x, (0, 0, r, r), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.unsqueeze(3))
    return x


def random_block_scores(
    batch: int,
    grid_h: int,
    grid_w: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """One fixed random atom ranking shared by static/adaptive controls."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) + 104729)
    return torch.rand(batch, 1, grid_h, grid_w, generator=gen).to(device)


def _active_schedule(
    num_steps: int,
    budget: int,
    rounds: int,
    window: tuple[float, float],
) -> dict[int, int]:
    """Unique acquisition schedule that cannot silently drop budget."""
    if budget <= 0:
        return {}
    n_rounds = max(1, int(rounds))
    lo = min(num_steps - 1, max(0, int(round(window[0] * (num_steps - 1)))))
    hi = min(num_steps - 1, max(lo, int(round(window[1] * (num_steps - 1)))))
    candidates = list(range(lo, hi + 1))
    if n_rounds > len(candidates):
        raise ValueError(
            f"{n_rounds} rounds do not fit unique acquisition window [{lo}, {hi}]"
        )
    if n_rounds == 1:
        steps = [lo]
    else:
        steps = [
            candidates[round(i * (len(candidates) - 1) / (n_rounds - 1))]
            for i in range(n_rounds)
        ]
    if len(set(steps)) != len(steps):
        raise AssertionError(f"duplicate acquisition steps: {steps}")
    base, extra = divmod(int(budget), n_rounds)
    schedule = {step: base + (1 if i < extra else 0)
                for i, step in enumerate(steps)}
    if sum(schedule.values()) != int(budget):
        raise AssertionError("active schedule lost measurement budget")
    return schedule


# ---------------------------------------------------------------------------
# Sampler: DDIM + (PV-gated) data consistency + active acquisition
# ---------------------------------------------------------------------------

@dataclass
class PVASConfig:
    num_steps: int = 100
    eta: float = 0.0                    # DDIM stochasticity
    pv_eps_probe: float = 5e-2
    pv_n_probes: int = 1
    pv_every: int = 5                   # refresh PV every k steps (cost control)
    pv_smooth_sigma: float = 1.0
    use_pv_gate: bool = True            # PV-gated vs scalar PiGDM-style gain
    active_rounds: int = 2
    active_window: tuple[float, float] = (0.45, 0.75)  # fraction of trajectory
    policy: str = "pv"                  # pv|operator_diag|sketch_*|baselines
    ensemble_k: int = 8                 # posterior samples for 'ensemble' policy
    sketch_rank: int = 4
    sketch_fd_step: float = 5e-2
    sketch_psd_rtol: float = 1e-6
    sketch_noise_variance_floor: float = 1e-4
    seed: int = 0


@dataclass
class PVASResult:
    x0: torch.Tensor
    state: BlockMaskState
    nfe: int
    pv_maps: list = field(default_factory=list)
    acq_steps: list = field(default_factory=list)


def _dc_update(x0hat: torch.Tensor, state: BlockMaskState, gain: torch.Tensor) -> torch.Tensor:
    """Kalman-style data consistency: optional LR box-average DC, then block DC."""
    if state.y_lr is not None:
        b = state.block
        resid = state.y_lr - torch.nn.functional.avg_pool2d(x0hat, b)
        x0hat = x0hat + torch.nn.functional.interpolate(resid, scale_factor=b,
                                                        mode="nearest")
    return x0hat + gain * state.mask * (state.y - x0hat * state.mask)


@torch.no_grad()
def run_pvas(den: EpsDenoiser, state: BlockMaskState, x0_full: torch.Tensor,
             budget_blocks: int, cfg: PVASConfig,
             ensemble_scorer=None) -> PVASResult:
    """Run DDIM posterior sampling with (optional) active block acquisition.

    x0_full is the held-out ground truth used ONLY to (a) simulate new
    measurements in the retrospective protocol and (b) implement the
    retrospective greedy-oracle reference. That reference is not a bound on
    final reconstruction quality. x0_full never enters the sampler otherwise.
    """
    device = x0_full.device
    gen_sample = torch.Generator(device="cpu").manual_seed(cfg.seed)
    gen_acquire = torch.Generator(device="cpu").manual_seed(cfg.seed + 2_000_003)
    B, C, H, W = x0_full.shape
    abar = den.abar
    T = len(abar)
    ts = ddim_timesteps(T, cfg.num_steps)
    xt = torch.randn(x0_full.shape, generator=gen_sample).to(device)
    state = state.clone()
    initial_blocks = state.n_blocks_observed.clone()

    # Acquisition schedule: rounds at fractions of the trajectory
    acq_schedule = _active_schedule(
        cfg.num_steps, budget_blocks, cfg.active_rounds, cfg.active_window
    )

    pv_cached = None
    result = PVASResult(x0=None, state=state, nfe=0)
    for i, t in enumerate(ts.tolist()):
        a_t = abar[t].to(device)
        x0hat, eps_theta = den.predict(xt, t)

        # --- posterior variance (refreshed every pv_every steps if needed) ---
        need_pv = (cfg.use_pv_gate or (i in acq_schedule and cfg.policy == "pv"))
        if need_pv and (pv_cached is None or i % cfg.pv_every == 0 or i in acq_schedule):
            # Counter-based seed: gate-on and selection-only arms use the same
            # probe prefix at the same diffusion timestep even if one arm
            # refreshed its gate at additional earlier steps.
            probe_seed = int(
                (int(cfg.seed) * 1_000_003 + int(t) * 104_729 + 17)
                % (2**63 - 1)
            )
            gen_probe = torch.Generator(device="cpu").manual_seed(probe_seed)
            pv_cached = pv_estimate(den, xt, t, cfg.pv_eps_probe, cfg.pv_n_probes,
                                    pool_block=None, smooth_sigma=cfg.pv_smooth_sigma,
                                    generator=gen_probe, clamp_final=False)
            if i in acq_schedule:
                result.pv_maps.append((t, pv_cached.detach().cpu()))

        # --- data consistency on x0hat ---
        sig2_t = ((1.0 - a_t) / a_t)
        sy2 = state.sigma_y ** 2
        if state.sigma_y == 0.0:
            gain = torch.ones(1, 1, 1, 1, device=device)
        elif cfg.use_pv_gate and pv_cached is not None:
            pv_gate = pv_cached.clamp_min(0.0)
            gain = pv_gate / (pv_gate + sy2)
        else:
            gain = (sig2_t / (sig2_t + sy2)).view(1, 1, 1, 1)
        x0hat = _dc_update(x0hat, state, gain)

        # --- active acquisition ---
        if i in acq_schedule:
            k = acq_schedule[i]
            scores = _score_blocks(cfg.policy, den, xt, t, x0hat, x0_full, state,
                                   pv_cached, cfg, gen_acquire, ensemble_scorer,
                                   k=k)
            state = acquire_blocks(state, x0_full, scores, k, gen_acquire)
            result.acq_steps.append((i, t, int(k)))
            # re-apply DC with the enlarged mask (raw x0hat from same eps)
            x0hat_raw = (xt - torch.sqrt(1.0 - a_t) * eps_theta) / torch.sqrt(a_t)
            x0hat = _dc_update(x0hat_raw, state, gain)

        # --- DDIM step (DDNM-style: keep the model's eps for the direction
        # term so the DC correction on x0hat propagates at full strength) ---
        if i + 1 < len(ts):
            t_next = ts[i + 1].item()
            a_n = abar[t_next].to(device)
            sigma = cfg.eta * torch.sqrt((1 - a_n) / (1 - a_t)) * torch.sqrt(1 - a_t / a_n)
            noise = torch.randn(xt.shape, generator=gen_sample).to(device)
            xt = (torch.sqrt(a_n) * x0hat
                  + torch.sqrt(torch.clamp(1 - a_n - sigma ** 2, min=0)) * eps_theta
                  + sigma * noise)
        else:
            xt = x0hat

    result.x0 = xt.clamp(-1, 1)
    result.state = state
    result.nfe = den.stats.nfe
    expected = torch.clamp(
        initial_blocks + int(budget_blocks),
        max=(H // state.block) * (W // state.block),
    )
    if not torch.equal(state.n_blocks_observed.cpu(), expected.cpu()):
        raise AssertionError(
            "active budget mismatch: "
            f"expected {expected.tolist()}, got {state.n_blocks_observed.tolist()}"
        )
    return result


def _score_blocks(policy: str, den: EpsDenoiser, xt: torch.Tensor, t: int,
                  x0hat: torch.Tensor, x0_full: torch.Tensor, state: BlockMaskState,
                  pv_map: torch.Tensor, cfg: PVASConfig, gen: torch.Generator,
                  ensemble_scorer, k: int | None = None) -> torch.Tensor:
    """Score candidate blocks for acquisition. Higher = acquired first."""
    B = xt.shape[0]
    block = state.block
    gh, gw = xt.shape[2] // block, xt.shape[3] // block
    if policy == "pv":
        # Aggregate the unbiased signed coordinate estimates into atoms before
        # the one positivity projection.
        return (
            torch.nn.functional.avg_pool2d(pv_map, block) * (block ** 2)
        ).clamp_min(0.0)
    if policy in {"operator_diag", "sketch_marginal", "sketch_logdet"}:
        from poc_natural.sketchsense import (
            average_pool_conditioning_gram,
            block_atom_ids,
            sketch_identity_measurements_batched,
        )
        from src.samplers.operator_sketch import (
            build_atom_information,
            conditional_atom_scores,
            diagonal_atom_scores,
            greedy_logdet_select,
        )

        probe_seed = int(
            (int(cfg.seed) * 1_000_003 + int(t) * 104_729 + 8_000_021)
            % (2**63 - 1)
        )
        probe_generator = torch.Generator(device="cpu").manual_seed(probe_seed)
        sketches = sketch_identity_measurements_batched(
            den,
            xt,
            t,
            num_probes=cfg.sketch_rank,
            fd_step=cfg.sketch_fd_step,
            generator=probe_generator,
            psd_rtol=cfg.sketch_psd_rtol,
        )
        atom_ids = block_atom_ids(
            xt.shape[1], xt.shape[2], xt.shape[3], block,
            device=xt.device,
        )
        num_atoms = gh * gw
        noise_variance = max(
            state.sigma_y ** 2,
            float(cfg.sketch_noise_variance_floor),
        )
        pooled_mask = (
            torch.nn.functional.avg_pool2d(state.mask, block)
            .flatten(1) > 0.5
        )
        score_batch = torch.full(
            (B, num_atoms),
            float("-inf"),
            device=xt.device,
            dtype=xt.dtype,
        )
        for image_index, sketch in enumerate(sketches):
            available = torch.nonzero(
                ~pooled_mask[image_index], as_tuple=False
            ).flatten()
            observed = torch.nonzero(
                pooled_mask[image_index], as_tuple=False
            ).flatten()
            initial_gram = (
                average_pool_conditioning_gram(
                    sketch,
                    block,
                    noise_variance,
                )
                if state.y_lr is not None
                else None
            )
            if policy == "operator_diag":
                score_batch[image_index] = diagonal_atom_scores(
                    sketch,
                    atom_ids,
                    noise_variance,
                    num_atoms=num_atoms,
                    complex_noise_is_total=False,
                    source="hutchinson",
                )
            else:
                information = build_atom_information(
                    sketch,
                    atom_ids,
                    noise_variance,
                    num_atoms=num_atoms,
                    complex_noise_is_total=False,
                )
                if policy == "sketch_marginal":
                    score_batch[image_index] = conditional_atom_scores(
                        information,
                        conditioned_on=observed,
                        initial_gram=initial_gram,
                    )
                else:
                    selected, _ = greedy_logdet_select(
                        information,
                        int(k or 0),
                        eligible=available,
                        conditioned_on=observed,
                        initial_gram=initial_gram,
                    )
                    if selected:
                        # acquire_blocks needs a score field; assigning a
                        # strict prefix order reproduces the joint-greedy set.
                        priority = torch.arange(
                            len(selected), 0, -1,
                            device=xt.device, dtype=xt.dtype,
                        )
                        score_batch[
                            image_index,
                            torch.as_tensor(selected, device=xt.device),
                        ] = priority
        return score_batch.view(B, 1, gh, gw)
    if policy == "random":
        return random_block_scores(B, gh, gw, cfg.seed, xt.device)
    if policy == "raster":
        base = torch.arange(gh * gw, 0, -1, dtype=torch.float32).view(1, 1, gh, gw)
        return base.expand(B, 1, gh, gw).to(xt.device)
    if policy == "oracle":
        err = (x0hat - x0_full).pow(2).mean(dim=1, keepdim=True)
        return torch.nn.functional.avg_pool2d(err, block) * (block ** 2)
    if policy == "ensemble":
        # AdaSense/ADS-style: variance across K posterior samples (expensive).
        assert ensemble_scorer is not None
        return ensemble_scorer(state, cfg)
    raise ValueError(policy)
