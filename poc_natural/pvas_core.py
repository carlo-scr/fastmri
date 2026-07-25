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
  diag_est = sigma_t^2 * (J v) * v   (unbiased), or the positive
  rank-1 form sigma_t^2 * (J v)^2 (biased by off-diagonal energy but positive;
  matches the MRI implementation).  Both are exposed; experiments use (J v)*v
  averaged over probes, floored at 0, then block-pooled.  The floor at 0
  introduces a small positive bias (only the pre-floor estimate is unbiased),
  traded for lower variance in the block-summed score.
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
    """
    mask: torch.Tensor
    y: torch.Tensor
    sigma_y: float
    block: int
    y_lr: torch.Tensor | None = None

    def clone(self) -> "BlockMaskState":
        return BlockMaskState(self.mask.clone(), self.y.clone(), self.sigma_y,
                              self.block,
                              None if self.y_lr is None else self.y_lr.clone())

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
    y = (x0 + sigma_y * noise) * mask
    y_lr = torch.nn.functional.avg_pool2d(x0, block) if with_lr else None
    return BlockMaskState(mask=mask, y=y, sigma_y=sigma_y, block=block, y_lr=y_lr)


def acquire_blocks(state: BlockMaskState, x0: torch.Tensor, scores: torch.Tensor,
                   k: int, gen: torch.Generator) -> BlockMaskState:
    """Acquire the top-k UNOBSERVED blocks by `scores` (B,1,gh,gw); returns new state.

    Retrospective protocol: new measurements are read off the held-out ground
    truth x0 (plus fresh noise), exactly as in retrospective active MRI.
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
    noise = torch.randn(x0.shape, generator=gen).to(x0.device)
    y_new = (x0 + state.sigma_y * noise) * new_mask
    mask = torch.clamp(state.mask + new_mask, 0, 1)
    y = state.y + y_new * (1 - (state.mask > 0.5).float())  # don't double-write
    return BlockMaskState(mask=mask, y=y, sigma_y=state.sigma_y, block=block,
                          y_lr=state.y_lr)


# ---------------------------------------------------------------------------
# Posterior-variance estimator (Tweedie second moment, Hutchinson probe)
# ---------------------------------------------------------------------------

@torch.no_grad()
def pv_estimate(den: EpsDenoiser, xt: torch.Tensor, t: int,
                eps_probe: float = 5e-2, n_probes: int = 1,
                pool_block: int | None = None,
                smooth_sigma: float | None = 1.0,
                mode: str = "jv_v") -> torch.Tensor:
    """Estimate diag Cov[x0 | x_t] per pixel (averaged over channels).

    mode "jv_v":  sigma_t^2 * (Jv) * v  -- unbiased for the diagonal.
    mode "jv_sq": sigma_t^2 * (Jv)^2    -- positive, biased (MRI-style rank-1).

    The probe is applied in the xtilde = x_t / sqrt(abar) domain, i.e. we
    perturb x_t by eps * sqrt(abar_t) * v so that the finite difference
    approximates (d mu / d xtilde) v.  Central differences cancel the O(eps)
    Taylor term, leaving O(eps^2) truncation error.  Returns (B,1,H,W) after
    channel-averaging (with a floor at 0: slightly positively biased, lower
    variance), optional Gaussian smoothing, and optional block sum-pooling
    (B,1,gh,gw).
    """
    a = den.abar[t].to(xt.device)
    sig2 = ((1.0 - a) / a).item()
    scale = eps_probe * math.sqrt(a.item())
    est = torch.zeros_like(xt)
    for _ in range(n_probes):
        v = torch.randn_like(xt)
        mu_p = den.x0hat(xt + scale * v, t)
        mu_m = den.x0hat(xt - scale * v, t)
        jv = (mu_p - mu_m) / (2.0 * eps_probe)
        est = est + (jv * v if mode == "jv_v" else jv.pow(2))
    est = sig2 * est / n_probes
    est = est.clamp_min(0.0).mean(dim=1, keepdim=True)  # channel-avg, floor at 0
    if smooth_sigma is not None and smooth_sigma > 0:
        est = _gaussian_blur(est, smooth_sigma)
    if pool_block is not None:
        est = torch.nn.functional.avg_pool2d(est, pool_block) * (pool_block ** 2)
    return est


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    x = torch.nn.functional.conv2d(x, k.unsqueeze(2), padding=(0, r))
    x = torch.nn.functional.conv2d(x, k.unsqueeze(3), padding=(r, 0))
    return x


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
    policy: str = "pv"                  # pv|random|raster|oracle|ensemble
    ensemble_k: int = 8                 # posterior samples for 'ensemble' policy
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
    measurements in the retrospective protocol and (b) implement the 'oracle'
    upper-bound policy. It never enters the sampler otherwise.
    """
    device = x0_full.device
    gen = torch.Generator().manual_seed(cfg.seed)
    B, C, H, W = x0_full.shape
    abar = den.abar
    T = len(abar)
    ts = ddim_timesteps(T, cfg.num_steps)
    xt = torch.randn(x0_full.shape, generator=gen).to(device)
    state = state.clone()

    # Acquisition schedule: rounds at fractions of the trajectory
    n_rounds = cfg.active_rounds if budget_blocks > 0 else 0
    if n_rounds > 0:
        fr = torch.linspace(cfg.active_window[0], cfg.active_window[1], n_rounds)
        acq_at = set((fr * (cfg.num_steps - 1)).round().long().tolist())
        per_round = [budget_blocks // n_rounds + (1 if i < budget_blocks % n_rounds else 0)
                     for i in range(n_rounds)]
    else:
        acq_at, per_round = set(), []

    pv_cached = None
    result = PVASResult(x0=None, state=state, nfe=0)
    round_idx = 0

    for i, t in enumerate(ts.tolist()):
        a_t = abar[t].to(device)
        x0hat, eps_theta = den.predict(xt, t)

        # --- posterior variance (refreshed every pv_every steps if needed) ---
        need_pv = (cfg.use_pv_gate or (i in acq_at and cfg.policy == "pv"))
        if need_pv and (pv_cached is None or i % cfg.pv_every == 0 or i in acq_at):
            pv_cached = pv_estimate(den, xt, t, cfg.pv_eps_probe, cfg.pv_n_probes,
                                    pool_block=None, smooth_sigma=cfg.pv_smooth_sigma)
            if i in acq_at:
                result.pv_maps.append((t, pv_cached.detach().cpu()))

        # --- data consistency on x0hat ---
        sig2_t = ((1.0 - a_t) / a_t)
        sy2 = state.sigma_y ** 2
        if state.sigma_y == 0.0:
            gain = torch.ones(1, 1, 1, 1, device=device)
        elif cfg.use_pv_gate and pv_cached is not None:
            gain = pv_cached / (pv_cached + sy2)
        else:
            gain = (sig2_t / (sig2_t + sy2)).view(1, 1, 1, 1)
        x0hat = _dc_update(x0hat, state, gain)

        # --- active acquisition ---
        if i in acq_at and round_idx < len(per_round):
            k = per_round[round_idx]
            round_idx += 1
            scores = _score_blocks(cfg.policy, den, xt, t, x0hat, x0_full, state,
                                   pv_cached, cfg, gen, ensemble_scorer)
            state = acquire_blocks(state, x0_full, scores, k, gen)
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
            noise = torch.randn(xt.shape, generator=gen).to(device)
            xt = (torch.sqrt(a_n) * x0hat
                  + torch.sqrt(torch.clamp(1 - a_n - sigma ** 2, min=0)) * eps_theta
                  + sigma * noise)
        else:
            xt = x0hat

    result.x0 = xt.clamp(-1, 1)
    result.state = state
    result.nfe = den.stats.nfe
    return result


def _score_blocks(policy: str, den: EpsDenoiser, xt: torch.Tensor, t: int,
                  x0hat: torch.Tensor, x0_full: torch.Tensor, state: BlockMaskState,
                  pv_map: torch.Tensor, cfg: PVASConfig, gen: torch.Generator,
                  ensemble_scorer) -> torch.Tensor:
    """Score candidate blocks for acquisition. Higher = acquired first."""
    B = xt.shape[0]
    block = state.block
    gh, gw = xt.shape[2] // block, xt.shape[3] // block
    if policy == "pv":
        return torch.nn.functional.avg_pool2d(pv_map, block) * (block ** 2)
    if policy == "random":
        return torch.rand(B, 1, gh, gw, generator=gen).to(xt.device)
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
