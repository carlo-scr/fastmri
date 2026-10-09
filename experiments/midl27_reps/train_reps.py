"""Repetition score matching: post-train an EDM prior on pairs of noisy repetitions.

For repetitions x_r = x + n_r, x_s = x + n_s with independent noise, train
D(x_r + sigma z; sigma_eff) to predict x_s. The optimum is E[x | input], the
clean posterior mean, because n_s is independent of the input. The input
already carries n_r, so the network is conditioned on the total noise level
sigma_eff = sqrt(floor^2 + sigma^2) and is only defined for sigma_eff >= floor.

Noise equalisation: the coil-combined noise std ``sig`` varies in space; we
add complementary noise so the total is exactly ``floor`` everywhere (also in
the background), which makes sigma_eff exact for every pixel.
"""
from __future__ import annotations

import copy
import math
import os
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data: all repetition images on the device, sampled as (input, target) pairs
# ---------------------------------------------------------------------------

class RepPairs:
    def __init__(self, slices: list[dict], device):
        imgs, sigs, floors, pairs = [], [], [], []
        for i, s in enumerate(slices):
            base = len(imgs)
            for r in range(s["x"].shape[0]):
                imgs.append(s["x"][r])
            n = s["x"].shape[0]
            pairs += [(i, base + r, base + q) for r in range(n) for q in range(n) if r != q]
            sigs.append(s["sig"])
            floors.append(s["floor"])
        self.X = torch.from_numpy(np.stack(imgs)).to(device)              # (N_img, H, W) complex64
        self.sig = torch.from_numpy(np.stack(sigs)).to(device)            # (N_slice, H, W)
        self.floor = torch.tensor(floors, dtype=torch.float32, device=device)
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)  # (N_pair, 3)
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def make(self, idx: torch.Tensor, sigma_eff: torch.Tensor, gen: torch.Generator,
             flip: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        """Inputs at total noise sigma_eff (B,), and Noise2Noise targets. Real 2-channel."""
        sl, a, b = self.pairs[idx].T
        f = torch.maximum(self.floor[sl], torch.zeros_like(self.floor[sl]))
        se = torch.maximum(sigma_eff, f)
        comp = torch.sqrt(torch.clamp(f[:, None, None] ** 2 - self.sig[sl] ** 2, min=0))   # (B,H,W)
        extra = torch.sqrt(torch.clamp(se ** 2 - f ** 2, min=0))[:, None, None]
        shape = (len(idx), 2) + tuple(self.X.shape[1:])
        z1 = torch.randn(shape, generator=gen, device=self.device)
        z2 = torch.randn(shape, generator=gen, device=self.device)
        xa, xb = self.X[a], self.X[b]
        inp = torch.stack([xa.real, xa.imag], 1) + comp[:, None] * z1 + extra[:, None] * z2
        tgt = torch.stack([xb.real, xb.imag], 1)
        if flip:
            fy = torch.rand(len(idx), generator=gen, device=self.device) < 0.5
            fx = torch.rand(len(idx), generator=gen, device=self.device) < 0.5
            inp = torch.where(fy[:, None, None, None], inp.flip(-2), inp)
            tgt = torch.where(fy[:, None, None, None], tgt.flip(-2), tgt)
            inp = torch.where(fx[:, None, None, None], inp.flip(-1), inp)
            tgt = torch.where(fx[:, None, None, None], tgt.flip(-1), tgt)
        return inp, tgt, se


def edm_weight(sigma: torch.Tensor, sigma_data: float = 0.5) -> torch.Tensor:
    return (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2


# ---------------------------------------------------------------------------
# Unbiased denoising comparison on held-out pairs
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoising_eval(nets: dict, data: RepPairs, sigmas: list[float], n_pairs: int = 256,
                   batch: int = 8, seed: int = 0) -> list[dict]:
    """Noise2Noise MSE per net and sigma_eff on the same inputs.

    E|D(input) - x_s|^2 = E|D(input) - x|^2 + E|n_s|^2, and the second term is the
    same for every net, so differences between nets are unbiased differences in
    true denoising error. Sigmas below a slice's floor are clamped to the floor.
    """
    g = torch.Generator(device=data.device).manual_seed(seed)
    idx_all = torch.randperm(len(data), generator=g, device=data.device)[:n_pairs]
    out = []
    for s in sigmas:
        acc = {k: 0.0 for k in nets}
        n = 0
        for i in range(0, len(idx_all), batch):
            idx = idx_all[i:i + batch]
            gi = torch.Generator(device=data.device).manual_seed(seed * 1000 + i)
            inp, tgt, se = data.make(idx, torch.full((len(idx),), s, device=data.device), gi, flip=False)
            for k, net in nets.items():
                acc[k] += float(((net(inp, se) - tgt) ** 2).mean(dim=(1, 2, 3)).sum())
            n += len(idx)
        out.append(dict(sigma=s, **{k: v / n for k, v in acc.items()}))
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def finetune(net: torch.nn.Module, data: RepPairs, *, steps: int, batch: int = 8, lr: float = 2e-5,
             ema_halflife: float = 500.0, warmup: int = 200, p_mean: float = -1.2, p_std: float = 1.2,
             out_dir: str | None = None, ckpt_every: int = 1000, log_every: int = 100,
             amp: bool = True, seed: int = 0, sync=None) -> tuple[torch.nn.Module, list[dict]]:
    """Full fine-tuning with EMA (LoRA is an ablation for the full study). Resumable from
    out_dir/ckpt.pt. Returns the EMA network and the loss history."""
    device = data.device
    net = net.train().requires_grad_(True)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    step, hist = 0, []
    ckpt = os.path.join(out_dir, "ckpt.pt") if out_dir else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=device)
        net.load_state_dict(st["net"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        step, hist = st["step"], st["hist"]
        print(f"[finetune] resumed at step {step}")
    g = torch.Generator(device=device).manual_seed(seed + step)
    beta = 0.5 ** (1.0 / ema_halflife)
    use_amp = amp and torch.device(device).type == "cuda"
    t0, run = time.time(), []
    while step < steps:
        idx = torch.randint(0, len(data), (batch,), generator=g, device=device)
        sig = torch.exp(p_mean + p_std * torch.randn(batch, generator=g, device=device))
        inp, tgt, se = data.make(idx, sig, g)
        for pg in opt.param_groups:
            pg["lr"] = lr * min(1.0, (step + 1) / warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            D = net(inp, se)
        loss = (edm_weight(se)[:, None, None, None] * (D.float() - tgt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn.detach(), 1 - beta)
        run.append(float(loss.detach()))
        step += 1
        if step % log_every == 0 or step == steps:
            hist.append(dict(step=step, loss=float(np.mean(run)), lr=opt.param_groups[0]["lr"],
                             min_per_1k=(time.time() - t0) / 60 / max(len(run), 1) * 1000))
            print(f"[finetune] step {step}/{steps} loss {hist[-1]['loss']:.4f} "
                  f"({hist[-1]['min_per_1k']:.1f} min/1k steps)")
            run, t0 = [], time.time()
        if ckpt and (step % ckpt_every == 0 or step == steps):
            tmp = ckpt + ".tmp"
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            step=step, hist=hist), tmp)
            os.replace(tmp, ckpt)
            if sync:
                sync()
    net.eval().requires_grad_(False)
    return ema, hist


# ---------------------------------------------------------------------------
# Sampling-time denoisers
# ---------------------------------------------------------------------------

class TwoRegime:
    """Post-trained denoiser above the floor, frozen 3 T prior below it."""

    def __init__(self, above, below, floor: float):
        self.above, self.below, self.floor = above, below, floor

    @property
    def nfe(self):
        return self.above.nfe + self.below.nfe

    def __call__(self, x, sigma_t):
        return (self.above if sigma_t >= self.floor else self.below)(x, sigma_t)


def truncate_schedule(sigmas: torch.Tensor, floor: float) -> torch.Tensor:
    """Keep steps with sigma >= floor; the sampler's last step then returns its x0 estimate."""
    keep = sigmas[sigmas >= floor]
    return keep if len(keep) else sigmas[:1]
