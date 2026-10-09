"""Frequency-resolved data consistency (FA-KGD, MIDL 2027 framing).

One module, one idea: every sampler shares the *same* data-consistency (DC)
solver, and "isotropic" vs. "+FA-KGD" differ ONLY in the per-frequency
posterior variance p_i(t) that the solver uses.

Model (prewhitened, normalised units; complex variances throughout):

    y_c = M F (S_c x) + n_c,      n_c ~ CN(0, s2)              (paper Eq. 1)
    x_0 | x_t  ~  CN(x0hat, P_t), P_t = F^H diag(p(t)) F        (paper Eq. 3)
    p_i(t) = kappa * S_i v_t / (S_i + v_t),   v_t = 2 sigma_t^2 (EDM: sigma_t^2
                                               per real channel)

    S_i = inf        -> IK-DC isotropic  (p = kappa v_t)
    S_i = sigma_d^2  -> white spectrum   (PiGDM's r_t^2 for data variance sigma_d^2)
    S_i = S_hat(rho) -> FA-KGD           (spectrum measured from the scan itself)

DC update (Kalman / MAP, paper Eqs. 2 and 6) solved by Fourier-preconditioned CG:

    (A^H A + s2 P^{-1}) delta = A^H (y - A x0hat),   x0tilde = x0hat + delta

Host samplers (paper Table 1) only differ in (a) how p is scaled,
(b) the number of CG iterations, (c) whether guidance goes through the
denoiser Jacobian (PiGDM, DPS), and (d) the re-noising rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# FFT helpers (centred, orthonormal -- identical to src.samplers.mri_forward)
# ---------------------------------------------------------------------------

def fft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def radius_grid(H: int, W: int, device=None) -> torch.Tensor:
    fy = torch.arange(H, dtype=torch.float32, device=device) - H // 2
    fx = torch.arange(W, dtype=torch.float32, device=device) - W // 2
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(gy ** 2 + gx ** 2)


def center_crop(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    ph, pw = max(0, h - H), max(0, w - W)
    if ph or pw:
        x = torch.nn.functional.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        H, W = x.shape[-2:]
    sh, sw = (H - h) // 2, (W - w) // 2
    return x[..., sh:sh + h, sw:sw + w]


# ---------------------------------------------------------------------------
# Noise: coil covariance Psi and prewhitening (Kellman & McVeigh 2005)
# ---------------------------------------------------------------------------

def estimate_psi(kspace_raw: np.ndarray, patch_frac: float = 0.06) -> np.ndarray:
    """Coil noise covariance from signal-poor k-space corners.

    ``kspace_raw``: (Nz, Nc, H, W) or (Nc, H, W) native (uncropped, readout-
    oversampled) k-space. Uses the four extreme corners of every slice and
    keeps the corner with the smallest trace (robust to one corner holding
    object energy). Returns Psi (Nc, Nc) complex Hermitian.

    fastMRI ships no noise prescans, so this background estimate is the
    standard substitute; it is a stated limitation in the paper.
    """
    k = kspace_raw if kspace_raw.ndim == 4 else kspace_raw[None]
    Nz, Nc, H, W = k.shape
    # fastMRI zero-pads the phase-encode (column) direction; padded columns
    # carry neither signal nor noise, so restrict to acquired columns.
    acquired = np.where(np.abs(k).sum(axis=(0, 1, 2)) > 0)[0]
    lo, hi = acquired.min(), acquired.max() + 1
    ph = max(4, int(round(patch_frac * H)))
    pw = max(4, int(round(patch_frac * (hi - lo))))
    corners = [k[:, :, :ph, lo:lo + pw], k[:, :, :ph, hi - pw:hi],
               k[:, :, -ph:, lo:lo + pw], k[:, :, -ph:, hi - pw:hi]]
    best, best_tr = None, np.inf
    for c in corners:
        v = np.moveaxis(c, 1, 0).reshape(Nc, -1)            # (Nc, samples)
        psi = (v @ v.conj().T) / v.shape[1]
        tr = float(np.real(np.trace(psi)))
        if tr < best_tr:
            best, best_tr = psi, tr
    return best


def whitening_matrix(psi: np.ndarray) -> np.ndarray:
    """W = L^{-1} with Psi = L L^H, so W Psi W^H = I."""
    L = np.linalg.cholesky(psi + 1e-12 * np.eye(psi.shape[0]) * np.real(np.trace(psi)))
    return np.linalg.inv(L)


def apply_whitening(k: torch.Tensor, Wm: torch.Tensor) -> torch.Tensor:
    """k: (Nc, H, W) complex -> W k along the coil axis."""
    Nc = k.shape[0]
    return (Wm @ k.reshape(Nc, -1)).reshape(k.shape)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def equispaced_mask(W: int, R: int, center_fraction: float, seed: int) -> torch.Tensor:
    """fastMRI-style equispaced Cartesian mask with a fully sampled centre.

    The spacing is adjusted so the total number of columns is ~W/R
    (as fastMRI's EquispacedMaskFractionFunc). Returns (W,) float.
    """
    rng = np.random.RandomState(seed)
    n_low = int(round(W * center_fraction))
    mask = np.zeros(W, dtype=np.float32)
    pad = (W - n_low + 1) // 2
    mask[pad:pad + n_low] = 1
    adjusted = (R * (n_low - W)) / (n_low * R - W) if n_low * R != W else R
    offset = rng.randint(0, max(1, round(adjusted)))
    acc = np.arange(offset, W - 1, adjusted)
    mask[np.around(acc).astype(np.int64)] = 1
    return torch.from_numpy(mask)


def random_mask(W: int, R: int, center_fraction: float, seed: int) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    n_low = int(round(W * center_fraction))
    mask = np.zeros(W, dtype=np.float32)
    c = W // 2
    mask[c - n_low // 2: c + n_low // 2] = 1
    prob = (W / R - n_low) / (W - n_low + 1e-8)
    mask = np.maximum(mask, (rng.uniform(size=W) < prob).astype(np.float32))
    return torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Sensitivities and the SENSE operator
# ---------------------------------------------------------------------------

def sens_maps_lowres(y: torch.Tensor, center_fraction: float, thresh: float = 0.02) -> torch.Tensor:
    """Low-resolution (ACS, Tukey-windowed) maps normalised so sum_c |S_c|^2 = 1
    inside the object; zero outside. Same estimator as src/samplers/sense.py."""
    Nc, H, W = y.shape
    aw = max(2, int(round(center_fraction * W)))
    ah = max(2, int(round(center_fraction * H)))
    sy, sx = H // 2 - ah // 2, W // 2 - aw // 2

    def tukey(n, a=0.5):
        x = torch.linspace(0, 1, n, device=y.device)
        w = torch.ones(n, device=y.device)
        e = a / 2
        l, r = x < e, x > 1 - e
        w[l] = 0.5 * (1 + torch.cos(math.pi * (2 * x[l] / a - 1)))
        w[r] = 0.5 * (1 + torch.cos(math.pi * (2 * x[r] / a - 2 / a + 1)))
        return w

    win = (tukey(ah)[:, None] * tukey(aw)[None, :]).to(y.dtype)
    acs = torch.zeros_like(y)
    acs[:, sy:sy + ah, sx:sx + aw] = y[:, sy:sy + ah, sx:sx + aw] * win
    lr = ifft2c(acs)
    rss = lr.abs().square().sum(0).sqrt()
    support = (rss > thresh * rss.max()).to(y.dtype)
    return lr / (rss + 1e-6) * support


def espirit_maps(y_full: torch.Tensor, calib_width: int = 24, thresh: float = 0.02,
                 crop: float = 0.95) -> torch.Tensor:
    """ESPIRiT maps (sigpy) from the central fully sampled columns of y_full.

    Falls back to low-resolution maps from the same columns if sigpy is
    unavailable (prints a warning; not for paper runs).
    """
    Nc, H, W = y_full.shape
    try:
        import sigpy.mri as mr
    except ImportError:
        print("[fdc] WARNING: sigpy missing -> low-resolution maps (pip install sigpy)")
        return sens_maps_lowres(y_full, calib_width / W)
    c = W // 2
    calib = y_full[..., c - calib_width // 2: c + calib_width // 2]
    k = torch.zeros_like(y_full)
    k[..., c - calib_width // 2: c + calib_width // 2] = calib
    maps = mr.app.EspiritCalib(k.cpu().numpy(), calib_width=calib_width, thresh=thresh,
                               crop=crop, show_pbar=False).run()
    return torch.from_numpy(maps).to(y_full.dtype)


class SenseOp:
    """A x = M F (S_c x);  A^H y = sum_c S_c^* F^H (M y_c)  (true adjoint)."""

    def __init__(self, mask_2d: torch.Tensor, sens: torch.Tensor):
        self.mask = mask_2d          # (H, W) real {0,1}
        self.sens = sens             # (Nc, H, W) complex

    def A(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = self.mask if mask is None else mask
        return m * fft2c(self.sens * x.unsqueeze(0))

    def AH(self, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        m = self.mask if mask is None else mask
        return (self.sens.conj() * ifft2c(m * y)).sum(0)


# ---------------------------------------------------------------------------
# Spectrum estimate S_hat(rho) from the measured data (paper Eq. 5)
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """Radial power spectrum S(rho) in normalised units (complex variance per
    Fourier coefficient). ``mode='powerlaw'`` (default) or ``'bins'``."""
    edges: np.ndarray
    binned: np.ndarray
    coef: tuple            # (log_a, b): S = exp(log_a) * rho^-b
    dc: float
    floor: float
    mode: str = "powerlaw"

    def __call__(self, rho: torch.Tensor) -> torch.Tensor:
        r = rho.clamp(min=1.0)
        if self.mode == "bins":
            centers = torch.tensor(0.5 * (self.edges[1:] + self.edges[:-1]), dtype=rho.dtype, device=rho.device)
            vals = torch.tensor(self.binned, dtype=rho.dtype, device=rho.device)
            idx = torch.bucketize(r, centers).clamp(1, len(centers) - 1)
            x0, x1 = centers[idx - 1], centers[idx]
            w = ((r - x0) / (x1 - x0)).clamp(0, 1)
            S = torch.exp((1 - w) * torch.log(vals[idx - 1]) + w * torch.log(vals[idx]))
        else:
            S = torch.exp(self.coef[0] - self.coef[1] * torch.log(r))
        S = torch.where(rho < 1.0, torch.full_like(S, self.dc), S)
        return S.clamp(min=self.floor)


def estimate_spectrum(y: torch.Tensor, mask_2d: torch.Tensor, s2: float,
                      n_bins: int = 48, mode: str = "powerlaw",
                      snr_min: float = 2.0) -> Spectrum:
    """S_i ~= sum_c |y_ci|^2 - Nc s2 on *sampled* frequencies, radially binned.

    Uses only acquired data. With sum_c|S_c|^2 = 1 on the object and smooth
    sensitivities, sum_c |y_ci|^2 approximates |(F x)_i|^2 + Nc s2.
    Power law fitted in log-log on bins whose power exceeds ``snr_min`` x the
    noise floor.
    """
    Nc, H, W = y.shape
    rho = radius_grid(H, W, device=y.device)
    pw = y.abs().square().sum(0)
    sel = mask_2d > 0
    r_s, p_s = rho[sel].cpu().numpy(), pw[sel].cpu().numpy()
    edges = np.linspace(0.0, float(rho.max()) + 1e-3, n_bins + 1)
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = (r_s >= edges[b]) & (r_s < edges[b + 1])
        if m.any():
            binned[b] = p_s[m].mean()
    noise = Nc * s2
    sig = binned - noise
    floor = max(1e-3 * s2, 1e-12)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(sig) & (sig > (snr_min - 1) * noise) & (centers >= 1.0)
    if ok.sum() < 3:
        ok = np.isfinite(sig) & (sig > 0) & (centers >= 1.0)
    b, log_a = np.polyfit(np.log(centers[ok]), np.log(sig[ok]), 1)
    sig_f = np.where(np.isfinite(sig) & (sig > floor), sig, np.nan)
    sig_f = np.where(np.isnan(sig_f), np.exp(log_a + b * np.log(np.maximum(centers, 1.0))), sig_f)
    dc_mask = r_s < 1.0
    dc = float(max(p_s[dc_mask].mean() - noise, floor)) if dc_mask.any() else float(sig_f[0])
    return Spectrum(edges=edges, binned=np.maximum(sig_f, floor), coef=(float(log_a), float(-b)),
                    dc=dc, floor=floor, mode=mode)


# ---------------------------------------------------------------------------
# Posterior variance p_i(t) and the shared DC solver
# ---------------------------------------------------------------------------

def posterior_variance(kind: str, sigma_t: float, rho: torch.Tensor, *,
                       spectrum: Optional[Spectrum] = None, data_var: float = 1.0,
                       kappa: float = 1.0) -> torch.Tensor:
    """p_i(t) on the full (H, W) grid.  kind in {'iso','white','spectral'}."""
    v = 2.0 * sigma_t ** 2
    if kind == "iso":
        return torch.full_like(rho, kappa * v)
    if kind == "white":
        return torch.full_like(rho, kappa * data_var * v / (data_var + v))
    if kind == "spectral":
        S = spectrum(rho)
        return kappa * S * v / (S + v)
    raise ValueError(kind)


def dc_solve(op: SenseOp, r: torch.Tensor, p: torch.Tensor, s2: float,
             mask: Optional[torch.Tensor] = None, n_iter: int = 10,
             tol: float = 1e-6, precond: bool = False) -> torch.Tensor:
    """Solve (A^H A + s2 P^{-1}) delta = A^H r with CG (warm start 0).

    P^{-1} = F^H diag(1/p) F. Optional Fourier-diagonal preconditioner
    F^H diag(1/(m_i + s2/p_i)) F is exact for single-coil data but HURTS on
    real ESPIRiT/SENSE data (it ignores coil coupling into unsampled lines),
    so it is off by default. Returns delta (H, W).
    """
    m = op.mask if mask is None else mask
    lam = s2 / p.clamp(min=1e-30)

    def Op(d):
        return op.AH(op.A(d, m), m) + ifft2c(lam * fft2c(d))

    def Minv(z):
        return ifft2c(fft2c(z) / (m + lam)) if precond else z

    b = op.AH(r, m)
    x = torch.zeros_like(b)
    res = b.clone()
    z = Minv(res)
    d = z.clone()
    rz = torch.vdot(res.flatten(), z.flatten()).real
    b_norm = torch.linalg.vector_norm(b).clamp(min=1e-30)
    for _ in range(n_iter):
        Ad = Op(d)
        alpha = rz / torch.vdot(d.flatten(), Ad.flatten()).real.clamp(min=1e-30)
        x = x + alpha * d
        res = res - alpha * Ad
        if torch.linalg.vector_norm(res) / b_norm < tol:
            break
        z = Minv(res)
        rz_new = torch.vdot(res.flatten(), z.flatten()).real
        d = z + (rz_new / rz.clamp(min=1e-30)) * d
        rz = rz_new
    return x


def dual_solve(op: SenseOp, r: torch.Tensor, p: torch.Tensor, s2: float,
               mask: Optional[torch.Tensor] = None, n_iter: int = 10,
               tol: float = 1e-6) -> torch.Tensor:
    """u = (A P A^H + s2 I)^{-1} r by CG in measurement space (PiGDM's form).

    Eigenvalues are >= s2, so truncation never divides by small p. Returns u
    (Nc, H, W); then A^H u is the PiGDM guidance vector and P A^H u = delta.
    """
    m = op.mask if mask is None else mask

    def Op(u):
        return op.A(ifft2c(p * fft2c(op.AH(u, m))), m) + s2 * u

    u = torch.zeros_like(r)
    res = r.clone()
    d = res.clone()
    rr = torch.vdot(res.flatten(), res.flatten()).real
    r_norm = torch.linalg.vector_norm(r).clamp(min=1e-30)
    for _ in range(n_iter):
        Ad = Op(d)
        alpha = rr / torch.vdot(d.flatten(), Ad.flatten()).real.clamp(min=1e-30)
        u = u + alpha * d
        res = res - alpha * Ad
        rr_new = torch.vdot(res.flatten(), res.flatten()).real
        if rr_new.sqrt() / r_norm < tol:
            break
        d = res + (rr_new / rr.clamp(min=1e-30)) * d
        rr = rr_new
    return u


# ---------------------------------------------------------------------------
# Optional components: frequency gate and residual refinement
# ---------------------------------------------------------------------------

def gate_mask(mask: torch.Tensor, rho: torch.Tensor, step: int, T: int,
              center_fraction: float, beta: float) -> torch.Tensor:
    """FPDC gate: keep sampled frequencies with rho <= r(t)."""
    r_max = float(rho.max())
    r_acs = center_fraction * r_max
    r_t = r_acs + (r_max - r_acs) * ((step + 1) / T) ** beta
    return mask * (rho <= r_t).to(mask.dtype)


class BandRefiner:
    """Band-wise moment correction of p (paper Sec. 3.4, 'residual refinement').

    Under the model, sum_c |r_ci|^2 ~ p_i + Nc s2 on sampled i. We track a
    multiplicative EMA factor c_b per radial band and return c_b * p_i.
    Also produces the calibration ratio used in Fig. 2 (right).
    """

    def __init__(self, rho: torch.Tensor, n_bands: int = 5, lam: float = 0.3,
                 clamp=(0.1, 10.0)):
        r_max = float(rho.max())
        self.band = torch.clamp((rho / r_max * n_bands).long(), max=n_bands - 1)
        self.n_bands, self.lam, self.clamp = n_bands, lam, clamp
        self.c = torch.ones(n_bands, device=rho.device)

    def ratio(self, r: torch.Tensor, p: torch.Tensor, s2: float, mask: torch.Tensor) -> torch.Tensor:
        Nc = r.shape[0]
        e = r.abs().square().sum(0)
        out = torch.full((self.n_bands,), float("nan"), device=r.device)
        for b in range(self.n_bands):
            sel = (self.band == b) & (mask > 0)
            if sel.any():
                out[b] = e[sel].mean() / (p[sel].mean() + Nc * s2)
        return out

    def update(self, r, p, s2, mask) -> torch.Tensor:
        ratio = self.ratio(r, p, s2, mask)
        ok = torch.isfinite(ratio)
        target = torch.where(ok, ratio.clamp(*self.clamp), self.c)
        self.c = (1 - self.lam) * self.c + self.lam * target
        return self.c[self.band] * p


# ---------------------------------------------------------------------------
# Sampler configuration
# ---------------------------------------------------------------------------

@dataclass
class MethodSpec:
    """One row of the paper tables.

    host:      'ikdc' | 'pigdm' | 'dds' | 'diffpir' | 'dps' | 'fakgd_v1'
    cov:       'iso' | 'white' | 'spectral'   (ignored by dps / fakgd_v1)
    kappa:     scale of p (the tuned 'step weight' of isotropic baselines)
    n_cg:      CG iterations of the DC solve
    eta:       re-noising stochasticity (1 = fresh noise, 0 = deterministic)
    gate_beta: None or FPDC exponent
    refine:    residual band refinement on/off
    guidance:  PiGDM/DPS guidance weight
    spec_mode: 'powerlaw' | 'bins' | 'oracle'
    s2_scale:  multiply the noise variance used by the solver (ablation h/i)
    """
    name: str
    host: str = "ikdc"
    cov: str = "iso"
    kappa: float = 1.0
    n_cg: int = 10
    eta: float = 1.0
    gate_beta: Optional[float] = None
    refine: bool = False
    guidance: float = 1.0
    spec_mode: str = "powerlaw"
    s2_scale: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def standard_methods() -> dict[str, MethodSpec]:
    """Default rows. Isotropic baselines get their kappa/eta from tuning."""
    M = {}
    for host, n_cg, eta in [("ikdc", 10, 1.0), ("pigdm", 10, 1.0), ("dds", 5, 0.15), ("diffpir", 10, 1.0)]:
        iso_cov = "white" if host == "pigdm" else "iso"
        M[f"{host}_iso"] = MethodSpec(f"{host}_iso", host=host, cov=iso_cov, n_cg=n_cg, eta=eta)
        M[f"{host}_fakgd"] = MethodSpec(f"{host}_fakgd", host=host, cov="spectral", n_cg=n_cg, eta=eta)
    M["dps"] = MethodSpec("dps", host="dps", guidance=1.0)
    M["fakgd_v1_clamp"] = MethodSpec("fakgd_v1_clamp", host="fakgd_v1", extra={"m_step_mode": "clamp"})
    M["fakgd_v1_off"] = MethodSpec("fakgd_v1_off", host="fakgd_v1", extra={"m_step_mode": "off"})
    M["fakgd_v1_full"] = MethodSpec("fakgd_v1_full", host="fakgd_v1", extra={"m_step_mode": "full"})
    return M


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

def _complex_randn(shape, gen, device):
    return torch.complex(torch.randn(shape, generator=gen, device=device),
                         torch.randn(shape, generator=gen, device=device))


def _denoise_with_vjp(denoiser, x_t: torch.Tensor, sigma_t: float):
    """Return x0hat and a closure computing J^H g (real-linear VJP)."""
    net = denoiser.net
    xr = torch.stack([x_t.real, x_t.imag]).unsqueeze(0).float().detach().requires_grad_(True)
    sig = torch.tensor([sigma_t], dtype=torch.float32, device=xr.device)
    out = net(xr, sig)
    x0 = (out[0, 0] + 1j * out[0, 1]).to(x_t.dtype)
    denoiser.nfe += 1

    def vjp(g: torch.Tensor) -> torch.Tensor:
        gr = torch.stack([g.real, g.imag]).unsqueeze(0).float()
        (grad,) = torch.autograd.grad(out, xr, grad_outputs=gr, retain_graph=False)
        return (grad[0, 0] + 1j * grad[0, 1]).to(x_t.dtype)

    return x0.detach(), vjp


def run_sampler(spec: MethodSpec, *, denoiser, op: SenseOp, y: torch.Tensor, s2: float,
                sigmas: torch.Tensor, center_fraction: float, spectrum: Optional[Spectrum] = None,
                data_var: float = 1.0, seed: int = 0, x_ref: Optional[torch.Tensor] = None,
                diagnostics: bool = False) -> dict:
    """Run one reconstruction. All tensors on the same device.

    y: (Nc, H, W) prewhitened, normalised, masked k-space.  s2: its noise variance.
    x_ref: complex reference image (normalised) for per-step diagnostics only.
    """
    device = y.device
    Nc, H, W = y.shape
    T = len(sigmas)
    rho = radius_grid(H, W, device=device)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x_t = float(sigmas[0]) * _complex_randn((H, W), gen, device) / math.sqrt(1.0)
    s2_solver = s2 * spec.s2_scale
    refiner = BandRefiner(rho) if (spec.refine or diagnostics) else None
    diag = {"band_nmse": [], "calib_ratio": [], "sigma": []}
    ref_k = fft2c(x_ref) if (diagnostics and x_ref is not None) else None

    for step in range(T):
        sigma_t = float(sigmas[step])
        sigma_next = float(sigmas[step + 1]) if step < T - 1 else 0.0
        mask = op.mask
        if spec.gate_beta is not None:
            mask = gate_mask(op.mask, rho, step, T, center_fraction, spec.gate_beta)

        needs_vjp = spec.host in ("pigdm", "dps")
        if needs_vjp:
            with torch.enable_grad():
                x0hat, vjp = _denoise_with_vjp(denoiser, x_t, sigma_t)
        else:
            x0hat = denoiser(x_t, sigma_t)
        r = y - op.A(x0hat, mask)

        if spec.host == "dps":
            res_norm = torch.linalg.vector_norm(r).clamp(min=1e-12)
            g = op.AH(r, mask)                      # -(1/2) grad of ||r||^2 w.r.t. x0hat
            with torch.enable_grad():
                step_dir = vjp(g)
            x0tilde = x0hat
            eps_hat = (x_t - x0hat) / sigma_t
            x_t = x0tilde + sigma_next * eps_hat if sigma_next > 0 else x0tilde
            x_t = x_t + spec.guidance * step_dir / res_norm
            continue

        p = posterior_variance(spec.cov, sigma_t, rho, spectrum=spectrum,
                               data_var=data_var, kappa=spec.kappa)
        if diagnostics and refiner is not None:
            diag["calib_ratio"].append(refiner.ratio(r, p, s2_solver, mask).tolist())
        if spec.refine:
            p = refiner.update(r, p, s2_solver, mask)

        if spec.host == "pigdm":
            # x0' = x0hat + w * v_t * J^H A^H (A P A^H + s2 I)^{-1} r   (ΠGDM with covariance P)
            u = dual_solve(op, r, p, s2_solver, mask=mask, n_iter=spec.n_cg)
            with torch.enable_grad():
                jg = vjp(op.AH(u, mask))
            x0tilde = x0hat + spec.guidance * (2.0 * sigma_t ** 2) * jg
        else:
            delta = dc_solve(op, r, p, s2_solver, mask=mask, n_iter=spec.n_cg)
            x0tilde = x0hat + delta

        if diagnostics and ref_k is not None:
            diag["band_nmse"].append(_band_nmse(fft2c(x0hat), ref_k, rho))
            diag["sigma"].append(sigma_t)

        if sigma_next <= 0:
            x_t = x0tilde
            break
        eps_hat = (x_t - x0tilde) / sigma_t
        z = _complex_randn((H, W), gen, device)
        if spec.host == "diffpir":
            zeta = spec.eta
            x_t = x0tilde + sigma_next * (math.sqrt(1 - zeta) * eps_hat + math.sqrt(zeta) * z)
        else:
            eta = spec.eta
            x_t = x0tilde + sigma_next * (math.sqrt(max(0.0, 1 - eta ** 2)) * eps_hat + eta * z)

    out = {"recon": x_t}
    if diagnostics:
        out["diagnostics"] = diag
    return out


def _band_nmse(est_k: torch.Tensor, ref_k: torch.Tensor, rho: torch.Tensor, n_bands: int = 5) -> list:
    r_max = float(rho.max())
    band = torch.clamp((rho / r_max * n_bands).long(), max=n_bands - 1)
    out = []
    for b in range(n_bands):
        sel = band == b
        out.append(float((est_k[sel] - ref_k[sel]).abs().square().sum()
                         / ref_k[sel].abs().square().sum().clamp(min=1e-30)))
    return out


# ---------------------------------------------------------------------------
# Original CS231N FA-KGD (for the C3 / M-step diagnosis), via src/samplers
# ---------------------------------------------------------------------------

def run_fakgd_v1(spec: MethodSpec, *, denoiser, op: SenseOp, y: torch.Tensor, s2: float,
                 sigmas: torch.Tensor, center_fraction: float, seed: int = 0) -> dict:
    from src.samplers.multicoil import run_fakgd_mc
    from src.samplers.sense import MultiCoilSENSE
    Nc, H, W = y.shape
    sig_c = torch.full((Nc,), float(s2), device=y.device)
    sop = MultiCoilSENSE(op.mask, op.sens, sig_c).to(y.device)
    rho = radius_grid(H, W, device=y.device)
    init = torch.full((H, W), float(s2), device=y.device)   # flat = white noise (real data)
    res = run_fakgd_mc(y, sop, sigmas, denoiser, init,
                       r_acs=center_fraction * float(rho.max()), r_max=float(rho.max()),
                       beta_fpdc=spec.extra.get("beta_fpdc", 1.0),
                       alpha_ema=spec.extra.get("alpha_ema", 0.95),
                       m_step_mode=spec.extra.get("m_step_mode", "clamp"), seed=seed)
    rel = float((res["sigma_i_sq_final"] - init).abs().mean() / init.abs().mean())
    return {"recon": res["recon"], "m_step_rel_change": rel}


# ---------------------------------------------------------------------------
# Metrics (volume-level aggregation in analysis.py)
# ---------------------------------------------------------------------------

def slice_metrics(rec_mag: torch.Tensor, ref_mag: torch.Tensor, data_range: float,
                  fg: Optional[torch.Tensor] = None, eval_size: int = 320) -> dict:
    from skimage.metrics import structural_similarity
    from scipy.ndimage import gaussian_laplace
    g = center_crop(ref_mag, eval_size, eval_size).float().cpu()
    r = center_crop(rec_mag, eval_size, eval_size).float().cpu()
    se = float((g - r).square().sum())
    out = {
        "sse": se, "ref_sq": float(g.square().sum()), "numel": int(g.numel()),
        "psnr": float(10 * math.log10(data_range ** 2 / max(se / g.numel(), 1e-20))),
        "nmse": se / max(float(g.square().sum()), 1e-20),
        "ssim": float(structural_similarity(g.numpy(), r.numpy(), data_range=data_range)),
    }
    lg, lr = gaussian_laplace(g.numpy(), 1.5), gaussian_laplace(r.numpy(), 1.5)
    out["hfen"] = float(np.linalg.norm(lg - lr) / max(np.linalg.norm(lg), 1e-20))
    if fg is not None:
        f = center_crop(fg, eval_size, eval_size).cpu().bool()
        out["sse_fg"] = float((g - r)[f].square().sum())
        out["numel_fg"] = int(f.sum())
    return out
