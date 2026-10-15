"""Exploration pilot: uncertainty, re-measurement, averaging and trust (shared machinery).

The pilot asks which decision-relevant signals a diffusion posterior gives on our data:
when to measure again (add k-space lines), when to measure longer (more averages), where to
measure, and when not to trust a reconstruction (failure / OOD monitoring). Notebook:
``uq_pilot_colab.ipynb`` (built by ``build_notebook.py``, which also embeds this file).

What this module adds to ``experiments/midl27/fdc.py`` and ``experiments/midl27_reps/m4raw.py``:
  * a batched IK-DC posterior sampler: K posterior samples plus F hold-out members in one batch;
  * exactly nested Cartesian masks, so "measure again" means adding lines to the ones already
    acquired; per-column averaging weights, so "measure longer" uses the same solver;
  * per-slice records: errors (noise-corrected on M4Raw, where the reference is a held-out
    repetition), ensemble spread, k-space hold-out residuals, the analytic frequency-aware
    (Wiener / FA-KGD) variance and value of information, prior typicality;
  * controlled shifts: added noise, rigid motion mid-scan, small synthetic lesions;
  * decision analyses: AUROC, re-acquisition cascades with Learn-then-Test thresholds,
    averaging allocation curves, conformal coverage.

Conventions follow fdc.py: centred orthonormal FFTs, prewhitened k-space (unit complex noise
variance per acquired sample, raw units), images normalised by an ACS-based scale, complex
variances (E|.|^2) everywhere, EDM noise levels per real channel.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "experiments" / "midl27"), str(ROOT / "experiments" / "midl27_reps")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fdc  # noqa: E402
import m4raw as M  # noqa: E402
from fdc import fft2c, ifft2c  # noqa: E402


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % (2 ** 31)


# ---------------------------------------------------------------------------
# Nested Cartesian masks
# ---------------------------------------------------------------------------

def line_order(acq: np.ndarray, n_acs: int, seed: int) -> np.ndarray:
    """Acquisition priority of phase-encode columns: the centred ACS block first, then
    coarse-to-fine equispaced levels (offsets from the centre divisible by 64, 32, 16, ...),
    in random order within a level. Every mask is a prefix of this order, so masks of
    different accelerations are exactly nested."""
    W = len(acq)
    c = W // 2
    acs = np.arange(c - n_acs // 2, c - n_acs // 2 + n_acs)
    acs = acs[(acs >= 0) & (acs < W)]
    acs = acs[acq[acs]]
    rest = np.setdiff1d(np.flatnonzero(acq), acs)
    d = np.abs(rest - c)
    level = np.array([(int(v) & -int(v)).bit_length() - 1 if v else 30 for v in d])
    rnd = np.random.default_rng(seed).random(len(rest))
    return np.concatenate([acs, rest[np.lexsort((rnd, -level))]]).astype(int)


def n_lines(acq: np.ndarray, R: float) -> int:
    n = int(acq.sum())
    return n if R <= 1 else max(1, int(round(n / R)))


# ---------------------------------------------------------------------------
# Cases: one slice with everything needed to simulate acquisitions and score them
# ---------------------------------------------------------------------------

@dataclass
class Case:
    key: str
    dataset: str            # 'fastmri' | 'm4raw'
    unit: str               # statistical unit: volume (fastMRI) or subject (M4Raw)
    contrast: str
    slice: int
    k: np.ndarray           # (rep, C, H, W) complex64, whitened k-space, raw units (rep = 1 for fastMRI)
    sens: np.ndarray        # (C, H, W) ESPIRiT maps
    acq: np.ndarray         # (W,) bool, columns that carry data
    scale: float            # normalisation (ACS 99th percentile, identical for every mask)
    n_acs: int
    region: np.ndarray      # (H, W) bool evaluation region
    support: np.ndarray     # (H, W) bool coil-map support
    ref: np.ndarray         # (H, W) complex reference, normalised
    ref_nv: np.ndarray      # (H, W) variance of ref around the truth (normalised)
    ref_exact: bool         # ref_nv unbiased per pixel (fastMRI: thermal; M4Raw: >= 3 repetitions)
    xs: np.ndarray | None = None   # (rep, H, W) MVUE of every repetition (M4Raw)
    v_rep: float = float("nan")    # region-mean variance of one repetition (thermal + physiological)
    meta: dict = field(default_factory=dict)

    @property
    def order(self) -> np.ndarray:
        return line_order(self.acq, self.n_acs, stable_seed(self.key, "order"))

    def columns(self, R: float) -> np.ndarray:
        return self.order[:n_lines(self.acq, R)]

    def acs_columns(self) -> np.ndarray:
        return self.order[:min(self.n_acs, int(self.acq.sum()))]

    def kspace(self, alloc: np.ndarray) -> np.ndarray:
        """(C, H, W): column w averaged over its first alloc[w] repetitions (rep 0 if alloc = 0)."""
        a = np.maximum(np.asarray(alloc, int), 1)
        out = np.empty(self.k.shape[1:], np.complex64)
        for n in np.unique(a):
            sel = a == n
            out[..., sel] = self.k[:n, ..., sel].mean(0)
        return out

    def light(self) -> "Case":
        """Copy without k-space and maps (for pickling summaries)."""
        return Case(**{**self.__dict__, "k": None, "sens": None})


def espirit(y_full: np.ndarray, calib_width: int, crop: float) -> np.ndarray:
    """ESPIRiT maps (sigpy) from the central calib_width x calib_width block. Runs on the GPU
    when cupy is available (Colab), else on the CPU. Same estimator as fdc.espirit_maps."""
    import sigpy as sp
    import sigpy.mri as mr
    devs = []
    try:
        import cupy  # noqa: F401
        devs.append(sp.Device(0))
    except Exception:  # noqa: BLE001
        pass
    devs.append(sp.cpu_device)
    for dev in devs:
        try:
            maps = mr.app.EspiritCalib(y_full.astype(np.complex64), calib_width=calib_width, thresh=0.02,
                                       crop=crop, device=dev, show_pbar=False).run()
            return np.asarray(sp.to_device(maps, sp.cpu_device)).astype(np.complex64)
        except Exception as e:  # noqa: BLE001
            if dev is devs[-1]:
                raise
            print(f"[uq] ESPIRiT on {dev} failed ({type(e).__name__}); falling back to CPU")


def load_fastmri(path: str, slice_ids, *, dataset: str = "fastmri", target=(384, 320),
                 calib_width: int = 24, crop: float = 0.95, acs_frac: float = 0.05) -> list[Case]:
    """Cases from one fastMRI multi-coil h5 (whitened with Psi from k-space corners, cropped to
    the prior's 384 x 320 grid, ESPIRiT maps, MVUE reference). Same conventions as
    experiments/midl27/run.py; slice_ids index the slices stored in the file."""
    import h5py
    with h5py.File(path, "r") as f:
        k = f["kspace"][()]
        src = f.attrs.get("midl27_source_slices")
        acqname = str(f.attrs.get("acquisition", ""))
    src = json.loads(src) if src is not None else list(range(k.shape[0]))
    psi = fdc.estimate_psi(k)
    Wm = torch.from_numpy(fdc.whitening_matrix(psi)).to(torch.complex64)
    vol = os.path.basename(path).replace(".h5", "")
    out = []
    for j in slice_ids:
        if j >= k.shape[0]:
            continue
        kw = fdc.apply_whitening(torch.from_numpy(k[j]).to(torch.complex64), Wm)
        coil = fdc.center_crop(ifft2c(kw), *target)
        y = fft2c(coil).numpy().astype(np.complex64)
        colE = (np.abs(y) ** 2).sum(axis=(0, 1))
        acq = colE > 1e-6 * np.median(colE)
        sens = espirit(y, calib_width, crop)
        x, den = M.mvue(y, sens)
        W = y.shape[-1]
        n_acs = max(8, int(round(acs_frac * W)))
        scale = M.acs_scale(y, n_acs)
        ref = (x / scale).astype(np.complex64)
        ref_nv = (M.mvue_noise_var(den, float(acq.mean())) / scale ** 2).astype(np.float32)
        region = M.support_region(ref, den)
        out.append(Case(key=f"{dataset}|{vol}|{src[j]}", dataset=dataset, unit=vol, contrast=acqname,
                        slice=int(src[j]), k=y[None], sens=sens, acq=acq, scale=scale, n_acs=n_acs,
                        region=region, support=den > 1e-6, ref=ref, ref_nv=ref_nv, ref_exact=True,
                        v_rep=float(ref_nv[region].mean()),
                        meta=dict(n_coils=int(y.shape[0]), psi_trace=float(np.real(np.trace(psi))))))
    return out


def retry(fn, tries: int = 6, wait: float = 5.0):
    """Call fn() until it succeeds, with exponential back-off. Zenodo answers 429/504 under load and
    sometimes drops a connection mid-file (http.client.IncompleteRead, which is not an OSError)."""
    import http.client
    for i in range(tries):
        try:
            return fn()
        except (OSError, http.client.HTTPException) as e:
            if i == tries - 1:
                raise
            print(f"[uq] {type(e).__name__}: {str(e)[:80]} -> retry in {wait * 2 ** i:.0f} s")
            time.sleep(wait * 2 ** i)


def prefetch(split: str, names: list[str], raw_dir: str, listing: list, workers: int = 4) -> None:
    """Range-fetch many M4Raw files in parallel (network-bound; m4raw.fetch skips complete files).
    Files that still fail after the parallel pass are retried one at a time before giving up."""
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def one(n):
        try:
            retry(lambda: M.fetch(split, [n], raw_dir, listing), tries=4)
            return None
        except Exception as e:  # noqa: BLE001
            return n, e

    with ThreadPoolExecutor(workers) as ex:
        failed = [r for r in ex.map(one, names) if r is not None]
    for n, e in failed:
        print(f"[uq] {n} failed in parallel ({type(e).__name__}); retrying alone")
        retry(lambda: M.fetch(split, [n], raw_dir, listing), tries=6, wait=10.0)
    print(f"[uq] fetched {len(names)} M4Raw files in {(time.time() - t0) / 60:.1f} min")


def load_m4raw(split: str, subject: str, contrast: str, cat: dict, listing: list, raw_dir: str, *,
               n_central: int = 6, pick=(1, 4), calib_width: int = 20, crop: float = 0.0,
               acs_frac: float = 0.05) -> list[Case]:
    """Cases from one M4Raw subject/contrast (range-fetched, registered, whitened with Psi from
    repetition differences, as experiments/midl27_reps/pilot.subject_kspace but only for the picked
    slices of the central n_central). Everything the reconstruction may use (maps, scale,
    measurements) comes from repetition 0. The reference is the mean of the held-out repetitions;
    its per-pixel variance is estimated from their spread (unbiased with >= 2 held-out
    repetitions, i.e. T1/T2 here), so per-pixel errors are unbiased without ground truth."""
    paths = M.fetch(split, cat[subject][contrast], raw_dir, listing)
    K, idx = M.load_reps(paths, n_central)
    pick = [z for z in pick if z < K.shape[1]]
    K, idx = K[:, pick], [idx[z] for z in pick]
    acq = M.acquired_columns(K)
    Kr, info = M.register_reps(K)
    Kw = M.whiten(Kr, M.psi_from_reps(Kr, acq))
    out = []
    for z in range(len(pick)):
        k = Kw[:, z].astype(np.complex64)
        sens = espirit(k[0], calib_width, crop)
        W = k.shape[-1]
        n_acs = max(8, int(round(acs_frac * W)))
        scale = M.acs_scale(k[0], n_acs)
        xs = np.stack([M.mvue(k[r] / scale, sens)[0] for r in range(k.shape[0])]).astype(np.complex64)
        den = (np.abs(sens) ** 2).sum(0)
        nv1 = M.mvue_noise_var(den, float(acq.mean())) / scale ** 2
        region = M.support_region(xs[0], den)
        v_rep = M.repetition_variance(list(xs), region)
        held = xs[1:]
        ref = held.mean(0)
        if len(held) >= 2:
            ref_nv, exact = (np.abs(held - ref) ** 2).sum(0) / (len(held) - 1) / len(held), True
        else:
            ref_nv, exact = nv1 * v_rep / max(float(nv1[region].mean()), 1e-12), False
        ri = [i for i in info if i["slice"] == z]
        out.append(Case(key=f"m4raw|{subject}|{contrast}|{idx[z]}", dataset="m4raw", unit=subject,
                        contrast=contrast, slice=int(idx[z]), k=k, sens=sens, acq=acq, scale=scale,
                        n_acs=n_acs, region=region, support=den > 1e-6, ref=ref.astype(np.complex64),
                        ref_nv=ref_nv.astype(np.float32), ref_exact=exact, xs=xs, v_rep=v_rep,
                        meta=dict(n_reps=int(k.shape[0]), nv1_mean=float(nv1[region].mean()),
                                  floor=float(np.sqrt(np.quantile(nv1[den > 1e-6], 0.99) / 2)),
                                  max_shift_px=float(max([abs(i["dy"]) + abs(i["dx"]) for i in ri] or [0])))))
    return out


# ---------------------------------------------------------------------------
# Controlled shifts
# ---------------------------------------------------------------------------

def random_motion(seed: int, amps=(1.0, 2.0, 4.0)) -> dict:
    """Rigid in-plane translation from a random point of a sequential phase-encode order onwards."""
    rng = np.random.default_rng(seed)
    a, th = float(rng.choice(amps)), float(rng.uniform(0, 2 * np.pi))
    return dict(t=float(rng.uniform(0.3, 0.7)), dy=a * math.sin(th), dx=a * math.cos(th), amp=a)


def make_lesions(case: Case, seed: int, n: int = 2, radius=(2.0, 4.0), contrast=(0.25, 0.6),
                 min_sep: float = 30.0, erode: int = 12) -> tuple[np.ndarray, list[dict]]:
    """Small Gaussian hyper-intensities inside the head, in tissue, with the local phase.
    Returns the complex image to add (normalised units) and per-lesion parameters."""
    from scipy.ndimage import binary_erosion, uniform_filter
    rng = np.random.default_rng(seed)
    inner = binary_erosion(case.region, iterations=erode)
    loc = uniform_filter(np.abs(case.ref), 7)
    locc = uniform_filter(case.ref.real, 7) + 1j * uniform_filter(case.ref.imag, 7)
    cand = np.argwhere(inner & (loc > np.quantile(loc[inner], 0.3))) if inner.any() else np.zeros((0, 2), int)
    H, W = case.ref.shape
    yy, xx = np.mgrid[:H, :W]
    blob, les = np.zeros((H, W), np.complex64), []
    for _ in range(200):
        if len(les) == n or len(cand) == 0:
            break
        cy, cx = cand[rng.integers(len(cand))]
        if any((cy - l["cy"]) ** 2 + (cx - l["cx"]) ** 2 < min_sep ** 2 for l in les):
            continue
        r = float(rng.uniform(*radius))
        s = r / 2.0
        amp = float(rng.uniform(*contrast)) * float(loc[cy, cx])
        prof = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2)).astype(np.float32)
        ph = np.exp(1j * np.angle(locc[cy, cx]))
        blob += (amp * ph * prof).astype(np.complex64)
        les.append(dict(cy=int(cy), cx=int(cx), radius=r, s=s, amp=amp, rel_contrast=amp / float(loc[cy, cx]),
                        prof=prof, phase=ph))
    return blob, les


# ---------------------------------------------------------------------------
# Batched SENSE operator, CG data consistency and the IK-DC posterior sampler
# ---------------------------------------------------------------------------

class BSense:
    """A x_b = w_b * F(S x_b),  A^H y_b = sum_c S_c^* F^H(w_b * y_bc).  w_b: (B, 1, W) column
    weights: 0 = not measured, sqrt(a) for a column averaged a times (whitens its noise to s2)."""

    def __init__(self, w: torch.Tensor, sens: torch.Tensor):
        self.w = w
        self.sens = sens

    def A(self, x: torch.Tensor) -> torch.Tensor:
        return self.w[:, None] * fft2c(self.sens[None] * x[:, None])

    def AH(self, y: torch.Tensor) -> torch.Tensor:
        return (self.sens.conj()[None] * ifft2c(self.w[:, None] * y)).sum(1)


def _bdot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.conj() * b).real.sum(dim=(-2, -1))


def dc_solve_b(op: BSense, r: torch.Tensor, p: torch.Tensor, s2: float, n_iter: int = 10) -> torch.Tensor:
    """Batched fdc.dc_solve: (A^H A + s2 P^-1) delta = A^H r per member (no preconditioner)."""
    lam = s2 / p.clamp(min=1e-30)

    def Op(d):
        return op.AH(op.A(d)) + ifft2c(lam * fft2c(d))

    b = op.AH(r)
    x = torch.zeros_like(b)
    res = b.clone()
    d = res.clone()
    rz = _bdot(res, res)
    for _ in range(n_iter):
        Ad = Op(d)
        alpha = rz / _bdot(d, Ad).clamp(min=1e-30)
        x = x + alpha[:, None, None] * d
        res = res - alpha[:, None, None] * Ad
        rz_new = _bdot(res, res)
        d = res + (rz_new / rz.clamp(min=1e-30))[:, None, None] * d
        rz = rz_new
    return x


def make_denoiser(net, below=None, floor: float = 0.0):
    """Complex (B, H, W) -> denoised. sigma: float (one level; below the floor `below` is used,
    for a post-trained prior) or a (B,) tensor (per-member levels, `net` only)."""
    def f(x: torch.Tensor, sigma) -> torch.Tensor:
        if torch.is_tensor(sigma):
            n, s = net, sigma.float().to(x.device)
        else:
            n = below if (below is not None and sigma < floor) else net
            s = torch.full((x.shape[0],), float(sigma), device=x.device)
        out = n(torch.stack([x.real, x.imag], 1).float(), s).float()
        return torch.complex(out[:, 0], out[:, 1])
    return f


@torch.no_grad()
def sample_batch(den_fn, op: BSense, y: torch.Tensor, s2: float, sigmas: torch.Tensor, *, cov: str = "iso",
                 kappa: float = 1.0, spectrum=None, n_cg: int = 10, seed: int = 0) -> torch.Tensor:
    """IK-DC with fresh re-noising (fdc host 'ikdc', eta = 1), batched over members.
    With B = 1 this reproduces fdc.run_sampler(ikdc) draw for draw."""
    B, C, H, W = y.shape
    dev = y.device
    rho = fdc.radius_grid(H, W, device=dev)
    g = torch.Generator(device=dev).manual_seed(int(seed))

    def crandn():
        return torch.complex(torch.randn((B, H, W), generator=g, device=dev),
                             torch.randn((B, H, W), generator=g, device=dev))

    x = float(sigmas[0]) * crandn()
    T = len(sigmas)
    for i in range(T):
        st = float(sigmas[i])
        sn = float(sigmas[i + 1]) if i < T - 1 else 0.0
        x0 = den_fn(x, st)
        p = fdc.posterior_variance(cov, st, rho, spectrum=spectrum, kappa=kappa)
        x0 = x0 + dc_solve_b(op, y - op.A(x0), p, s2, n_cg)
        if sn <= 0:
            return x0
        x = x0 + sn * crandn()
    return x


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def build_measurement(case: Case, *, R: float | None = None, cols=None, alloc=None, K: int = 8,
                      n_folds: int = 0, fold_frac: float = 0.1, noise_var: float = 0.0, motion: dict | None = None,
                      lesion: np.ndarray | None = None, device="cuda") -> dict:
    """Batch of K full-mask members plus n_folds hold-out members (each misses a disjoint ~fold_frac
    of the sampled non-ACS lines). alloc: per-column number of averages (overrides R / cols).
    noise_var: added complex variance per sample (raw units, single average). Shifts change the
    *measured* data only; the reference stays the clean object (plus the lesion, if any)."""
    C, H, W = case.k.shape[1:]
    if alloc is None:
        cols = case.columns(R) if cols is None else np.asarray(cols, int)
        alloc = np.zeros(W, int)
        alloc[cols] = 1
    alloc = np.asarray(alloc, int) * case.acq
    cols = np.flatnonzero(alloc)
    k = case.kspace(alloc).copy()
    if lesion is not None:
        kl = fft2c(torch.from_numpy(case.sens * (lesion * case.scale)[None])).numpy()
        k[..., case.acq] += kl[..., case.acq]
    if motion is not None:
        ky, kx = M._kgrid(H, W)
        ramp = np.exp(-2j * np.pi * (ky * motion["dy"] + kx * motion["dx"])).astype(np.complex64)
        ai = np.flatnonzero(case.acq)
        late = np.arange(W) >= ai[int(motion["t"] * (len(ai) - 1))]
        k[..., late] *= ramp[:, late]
    if noise_var > 0:
        rng = np.random.default_rng(stable_seed(case.key, "noise", noise_var))
        n = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)).astype(np.complex64) / np.sqrt(2)
        k[..., case.acq] += np.float32(np.sqrt(noise_var)) * n[..., case.acq]
    kn = (k / case.scale).astype(np.complex64)
    s2 = (1.0 + noise_var) / case.scale ** 2
    w = np.sqrt(alloc).astype(np.float32)
    masks, folds = [w] * K, []
    if n_folds:
        outer = np.setdiff1d(cols, case.acs_columns())
        perm = np.random.default_rng(stable_seed(case.key, "folds")).permutation(outer)
        nf = max(1, int(round(fold_frac * len(cols))))
        for f in range(n_folds):
            fc = perm[f * nf:(f + 1) * nf]
            if len(fc) == 0:
                break
            wf = w.copy()
            wf[fc] = 0
            masks.append(wf)
            folds.append(fc)
    wb = torch.from_numpy(np.stack(masks)).to(device)[:, None, :]
    kt = torch.from_numpy(kn).to(device)
    y = wb[:, None] * kt[None]
    op = BSense(wb, torch.from_numpy(case.sens).to(device))
    m2d = torch.from_numpy((alloc > 0).astype(np.float32)).to(device)[None, :].expand(H, W).contiguous()
    spec = fdc.estimate_spectrum(m2d * kt, m2d, s2)
    return dict(op=op, y=y, s2=s2, k_full=kt, alloc=alloc, cols=cols, folds=folds, spectrum=spec, m2d=m2d,
                K=K, B=len(masks), n_lines=int(len(cols)), cost=float(alloc.sum() / case.acq.sum()),
                R_eff=float(case.acq.sum() / max(len(cols), 1)))


# ---------------------------------------------------------------------------
# Prior typicality (multiscale denoising residual; MSMA / Graham et al. style)
# ---------------------------------------------------------------------------

TYP_SIGMAS = (0.02, 0.05, 0.1, 0.2, 0.5)


@torch.no_grad()
def typicality(den_fn, x: torch.Tensor, region: torch.Tensor, sigmas=TYP_SIGMAS, n: int = 2, seed: int = 0) -> list:
    """log of mean_region |D(x + s z, s) - x|^2 / (2 s^2) per noise level s."""
    dev = x.device
    H, W = x.shape
    g = torch.Generator(device=dev).manual_seed(seed)
    sig = torch.tensor(sigmas, device=dev, dtype=torch.float32).repeat_interleave(n)
    z = torch.complex(torch.randn((len(sig), H, W), generator=g, device=dev),
                      torch.randn((len(sig), H, W), generator=g, device=dev))
    d = den_fn(x[None] + sig[:, None, None] * z, sig)
    err = (d - x[None]).abs().square()[:, region].mean(1) / (2 * sig ** 2)
    return [float(v) for v in err.view(len(sigmas), n).mean(1).log()]


# ---------------------------------------------------------------------------
# Summaries of one ensemble
# ---------------------------------------------------------------------------

REL_EDGES = np.linspace(-7.0, 1.0, 33)     # log10(predicted patch MSE / signal power)
COV_EDGES = np.linspace(-5.0, 4.0, 91)     # log10(|xbar - ref|^2 / (pred + ref_nv)) per pixel


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def auroc(score, label) -> float:
    """Probability that a positive scores higher than a negative (ties count 1/2)."""
    from scipy.stats import rankdata
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _blocks(a: np.ndarray, b: int) -> np.ndarray:
    H, W = a.shape
    H2, W2 = H - H % b, W - W % b
    return a[:H2, :W2].reshape(H2 // b, b, W2 // b, b).mean(axis=(1, 3))


def _r(v, n=5):
    return [float(f"{x:.{n}g}") for x in np.asarray(v, float).ravel()]


@torch.no_grad()
def summarize(case: Case, X: torch.Tensor, meas: dict, *, ref=None, ref_nv=None, xs_last=None, den_fn=None,
              typ: bool = False, detail: bool = True, maps: bool = False, seed: int = 0):
    """Scalars (and small per-column / histogram arrays) describing one ensemble X (B, H, W).

    Errors are unbiased against the truth: mse = mean_region(|xbar - ref|^2 - ref_nv).
    pred = (1 + 1/K) x sample variance is the predicted MSE of the K-sample mean if the
    samples were exact posterior draws (calibration ratio mse / pred = 1)."""
    K, dev = meas["K"], X.device
    ref = case.ref if ref is None else ref
    ref_nv = case.ref_nv if ref_nv is None else ref_nv
    Rf = torch.from_numpy(np.ascontiguousarray(ref)).to(dev)
    NV = torch.from_numpy(np.ascontiguousarray(ref_nv, dtype=np.float32)).to(dev)
    G = torch.from_numpy(case.region).to(dev)
    S_ = torch.from_numpy(case.support).to(dev).float()
    H, W = ref.shape
    xk = X[:K]
    xbar = xk.mean(0)
    var = (xk - xbar).abs().square().sum(0) / max(K - 1, 1)
    pred = (1 + 1 / K) * var
    e = (xbar - Rf).abs().square() - NV

    def m(a):
        return float(a[G].mean())

    P = m(Rf.abs().square() - NV)
    rec = dict(K=K, sig_pow=P, ref_noise=m(NV), mse=m(e), pred_mse=m(pred), cost=meas["cost"],
               n_lines=meas["n_lines"], R_eff=meas["R_eff"], s2=meas["s2"])
    rec["nmse"], rec["pred_nmse"] = rec["mse"] / P, rec["pred_mse"] / P
    rec["psnr"] = float(10 * np.log10(float(Rf.abs()[G].max()) ** 2 / max(rec["mse"], 1e-12)))
    rec["mse_single"] = float(np.mean([m((xk[i] - Rf).abs().square() - NV) for i in range(K)]))
    XL = torch.from_numpy(xs_last).to(dev) if xs_last is not None else None
    for k in sorted({1, 2, 4, 8, 16, K}):
        if k > K:
            break
        xm = xk[:k].mean(0)
        rec[f"mse_k{k}"] = m((xm - Rf).abs().square() - NV)
        if k >= 2:
            rec[f"pred_k{k}"] = (1 + 1 / k) * m((xk[:k] - xm).abs().square().sum(0) / (k - 1))
        if XL is not None:
            rec[f"cr_k{k}"] = m((xm - XL).abs().square())
    # hold-out members: residual on the lines they did not see (k-space cross-validation)
    if meas["folds"]:
        res = noise = data = 0.0
        mse_h = []
        for f, fc in enumerate(meas["folds"]):
            xf = X[K + f]
            fm = torch.zeros(W, device=dev)
            fm[torch.as_tensor(fc, device=dev)] = 1
            yf = fm * meas["k_full"]
            res += float((yf - fm * fft2c(meas["op"].sens * xf[None])).abs().square().sum())
            noise += len(fc) * H * meas["k_full"].shape[0] * meas["s2"]
            data += float(yf.abs().square().sum())
            mse_h.append(m((xf - Rf).abs().square() - NV))
        rec.update(cv_ratio=res / noise, cv_nmse=(res - noise) / max(data - noise, 1e-30),
                   mse_holdout=float(np.mean(mse_h)))
    y0 = meas["m2d"] * meas["k_full"]
    nmeas = float(meas["m2d"].sum()) * meas["k_full"].shape[0]
    rec["snr_data"] = float(y0.abs().square().sum()) / (nmeas * meas["s2"]) - 1.0
    # analytic frequency-aware (Wiener / FA-KGD) posterior variance from the scan's own spectrum
    rho = fdc.radius_grid(H, W, device=dev)
    Sfull = meas["spectrum"](rho)
    s2 = meas["s2"]
    mcol = torch.from_numpy(meas["alloc"]).to(dev).float()[None, :].expand(H, W)
    s2_eff = s2 / mcol.clamp(min=1)
    Pi = torch.where(mcol > 0, Sfull * s2_eff / (Sfull + s2_eff), Sfull)
    Pi2 = torch.where(mcol > 0, Sfull * (s2_eff / 2) / (Sfull + s2_eff / 2), Sfull)
    Stot = float(Sfull.sum())
    rec["pred_ana_nmse"] = float(Pi.sum()) / Stot
    rec["voi_nex"] = float((Pi - Pi2).sum()) / Stot
    order = case.order
    nxt = order[meas["n_lines"]:min(len(order), 2 * meas["n_lines"])] if meas["n_lines"] < len(order) else []
    if len(nxt):
        nmask = torch.zeros(W, device=dev)
        nmask[torch.as_tensor(nxt, device=dev)] = 1
        rec["voi_next"] = float(((Sfull - Sfull * s2 / (Sfull + s2)) * nmask[None]).sum()) / Stot
    if detail:
        e_np, p_np, reg = e.cpu().numpy(), pred.cpu().numpy(), case.region
        mag = np.abs(xbar.cpu().numpy())
        gy, gx = np.gradient(mag)
        edge = gy ** 2 + gx ** 2
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(reg.ravel())
        sub = rng.choice(idx, min(20000, len(idx)), replace=False)
        rec["pix_rho_pred"] = _spearman(p_np.ravel()[sub], e_np.ravel()[sub])
        rec["pix_rho_edge"] = _spearman(edge.ravel()[sub], e_np.ravel()[sub])
        b = 8
        ok = _blocks(reg.astype(np.float32), b) >= 0.75
        eb, pb, edb, ib = (_blocks(a, b)[ok] for a in (e_np, p_np, edge, mag ** 2))
        rec.update(patch_rho_pred=_spearman(pb, eb), patch_rho_edge=_spearman(edb, eb),
                   patch_rho_int=_spearman(ib, eb))
        top = eb >= np.quantile(eb, 0.9)
        rec.update(patch_auc_pred=auroc(pb, top), patch_auc_edge=auroc(edb, top), patch_auc_int=auroc(ib, top))
        lb = np.clip(np.digitize(np.log10(np.maximum(pb / P, 1e-12)), REL_EDGES) - 1, 0, len(REL_EDGES) - 2)
        nb = len(REL_EDGES) - 1
        rec["rel_n"] = np.bincount(lb, minlength=nb).tolist()
        rec["rel_pred"] = _r(np.bincount(lb, weights=pb / P, minlength=nb))
        rec["rel_err"] = _r(np.bincount(lb, weights=eb / P, minlength=nb))
        sc = ((xbar - Rf).abs().square() / (pred + NV).clamp(min=1e-20))[G].cpu().numpy()
        rec["cov_hist"] = np.histogram(np.log10(np.maximum(sc, 1e-30)), COV_EDGES)[0].tolist()
        # per phase-encode column: ensemble variance, actual error, analytic variance
        Fx = fft2c(xk * S_)
        ensc = ((Fx - Fx.mean(0)).abs().square().sum(0) / max(K - 1, 1)).sum(0)
        errc = (fft2c(xbar * S_) - fft2c(Rf * S_)).abs().square().sum(0) - H * float((NV * S_).mean())
        Fc = fft2c(meas["op"].sens[None] * xk[:, None])
        vc = (Fc - Fc.mean(0)).abs().square().sum(0) / max(K - 1, 1)
        meas_cols = torch.from_numpy(meas["alloc"] > 0).to(dev)
        rec.update(col_err=_r(errc.cpu().numpy(), 4), col_ens=_r(ensc.cpu().numpy(), 4),
                   col_ana=_r(Pi.sum(0).cpu().numpy(), 4), col_coilvar=_r(vc.sum((0, 1)).cpu().numpy(), 4),
                   col_meas=(meas["alloc"] > 0).astype(int).tolist(), col_acq=case.acq.astype(int).tolist(),
                   disp_meas=float(vc[:, :, meas_cols].mean()) / s2 if bool(meas_cols.any()) else float("nan"))
    if typ and den_fn is not None:
        rec["typ_xbar"] = typicality(den_fn, xbar, G, seed=seed)
        rec["typ_ref"] = typicality(den_fn, Rf, G, seed=seed)
    mp = None
    if maps:
        mp = dict(xbar=xbar.cpu().numpy().astype(np.complex64), pred=pred.cpu().numpy().astype(np.float32),
                  err=e.cpu().numpy().astype(np.float32), ref=np.asarray(ref, np.complex64), region=case.region)
    return rec, mp


def lesion_metrics(case: Case, X_les: torch.Tensor, X_cln: torch.Tensor, K: int, blob: np.ndarray,
                   lesions: list[dict]) -> list[dict]:
    """Per lesion: contrast recovery (1 = fully reconstructed, 0 = removed), its spread over posterior
    samples, and whether the posterior variance rises where the lesion is."""
    dev = X_les.device
    ref_les = torch.from_numpy(case.ref + blob).to(dev)
    NV = torch.from_numpy(case.ref_nv).to(dev)
    xl, xc = X_les[:K], X_cln[:K]
    bl, bc = xl.mean(0), xc.mean(0)
    pl = (1 + 1 / K) * (xl - bl).abs().square().sum(0) / (K - 1)
    pc = (1 + 1 / K) * (xc - bc).abs().square().sum(0) / (K - 1)
    H, W = blob.shape
    yy, xx = np.mgrid[:H, :W]
    out = []
    for l in lesions:
        prof = torch.from_numpy(l["prof"]).to(dev)
        tmpl = prof * complex(l["amp"] * l["phase"])
        core = prof > 0.5
        d2 = (yy - l["cy"]) ** 2 + (xx - l["cx"]) ** 2
        ring = torch.from_numpy((d2 > (2.5 * l["s"]) ** 2) & (d2 < (5 * l["s"]) ** 2)).to(dev)
        den = float(tmpl.abs().square().sum())
        cr_s = [float((tmpl.conj() * (xl[i] - xc[i])).real.sum()) / den for i in range(K)]
        cr = float((tmpl.conj() * (bl - bc)).real.sum()) / den
        el = (bl - ref_les).abs().square() - NV
        out.append(dict(radius=l["radius"], rel_contrast=l["rel_contrast"], amp=l["amp"], cr=cr,
                        cr_sample_mean=float(np.mean(cr_s)), cr_sample_std=float(np.std(cr_s)),
                        unc_ratio_vs_clean=float(pl[core].mean() / pc[core].mean().clamp(min=1e-20)),
                        unc_ratio_vs_ring=float(pl[core].mean() / pl[ring].mean().clamp(min=1e-20)),
                        z_core=float(el[core].mean() / pl[core].mean().clamp(min=1e-20)),
                        z_ring=float(el[ring].mean() / pl[ring].mean().clamp(min=1e-20)),
                        err_core_rel=float(el[core].mean()) / l["amp"] ** 2,
                        unc_core_rel=float(pl[core].mean()) / l["amp"] ** 2))
    return out


# ---------------------------------------------------------------------------
# Resumable runner
# ---------------------------------------------------------------------------

class Runner:
    """Writes one JSONL record per (probe, case, tag); re-running skips existing records."""

    def __init__(self, path: str, den_fn, T: int, sigma_max: float = 10.0, device="cuda", maps_dir=None, sync=None):
        from src.samplers.schedules import edm_sigma_schedule
        self.path, self.den_fn, self.T, self.device = path, den_fn, T, device
        self.sigmas = edm_sigma_schedule(T, sigma_max=sigma_max).to(device)
        self.maps_dir, self.sync = maps_dir, sync
        self.done = set()
        if os.path.exists(path):
            for line in open(path):
                try:
                    self.done.add(json.loads(line)["rid"])
                except (json.JSONDecodeError, KeyError):
                    pass
        self.n_new, self.t0 = 0, time.time()

    def rid(self, probe, case, tag):
        return f"{probe}|{case.key}|{tag}|T{self.T}"

    def has(self, probe, case, tag):
        return self.rid(probe, case, tag) in self.done

    def write(self, rec: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.done.add(rec["rid"])
        self.n_new += 1

    def ensemble(self, case: Case, meas: dict, *, cov="iso", kappa=1.0, seed=0, den_fn=None) -> torch.Tensor:
        return sample_batch(den_fn or self.den_fn, meas["op"], meas["y"], meas["s2"], self.sigmas, cov=cov,
                            kappa=kappa, spectrum=meas["spectrum"], seed=seed)

    def run(self, probe, case: Case, tag, *, mkw: dict, cov="iso", kappa=1.0, skw: dict | None = None,
            extra: dict | None = None, den_fn=None, keep=False, maps=False):
        """mkw -> build_measurement, skw -> summarize. Returns (record, X if keep)."""
        rid = self.rid(probe, case, tag)
        if rid in self.done and not keep:
            return None, None
        t1 = time.time()
        meas = build_measurement(case, device=self.device, **mkw)
        seed = stable_seed(case.key, "sample", mkw.get("R"), mkw.get("K"))
        X = self.ensemble(case, meas, cov=cov, kappa=kappa, seed=seed, den_fn=den_fn)
        rec, mp = summarize(case, X, meas, den_fn=den_fn or self.den_fn, maps=maps, seed=seed, **(skw or {}))
        rec = dict(rid=rid, probe=probe, key=case.key, dataset=case.dataset, unit=case.unit, contrast=case.contrast,
                   slice=case.slice, tag=tag, R=mkw.get("R"), cov=cov, kappa=kappa, T=self.T, seed=seed,
                   v_rep=case.v_rep, ref_exact=case.ref_exact, runtime_s=time.time() - t1, **(extra or {}), **rec)
        if rid not in self.done:
            self.write(rec)
            if mp is not None and self.maps_dir:
                os.makedirs(self.maps_dir, exist_ok=True)
                np.savez_compressed(os.path.join(self.maps_dir, re.sub(r"[^A-Za-z0-9_.-]", "_", rid) + ".npz"), **mp)
        return rec, (X, meas) if keep else None

    def progress(self, label=""):
        print(f"[{label}] {self.n_new} new records, {(time.time() - self.t0) / 60:.1f} min")
        if self.sync:
            self.sync()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_jsonl(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])


def boot_ci(df, unit: str, fn, n: int = 1000, seed: int = 0, q=(0.025, 0.975)) -> tuple:
    """Percentile CI of fn(df) resampling statistical units with replacement."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    units = df[unit].unique()
    groups = {u: g for u, g in df.groupby(unit)}
    vals = []
    for _ in range(n):
        pick = rng.choice(units, len(units), replace=True)
        v = fn(pd.concat([groups[u] for u in pick]))
        if np.isfinite(v):
            vals.append(v)
    return (float(np.quantile(vals, q[0])), float(np.quantile(vals, q[1]))) if vals else (np.nan, np.nan)


def cascade_curve(Wd, levels, tau: float, score: str = "pred", thetas=None):
    """Re-acquisition cascade over nested levels (cheapest first). At each level the slice is
    accepted if score <= theta (always at the last level), otherwise the next level's extra lines
    are acquired. Returns cost (fraction of a full acquisition) and violation rate (err > tau)."""
    import pandas as pd
    if thetas is None:
        allv = np.concatenate([Wd[f"{score}_{L}"].to_numpy(float) for L in levels])
        thetas = np.unique(np.quantile(allv[np.isfinite(allv)], np.linspace(0, 1, 201)))
        thetas = np.concatenate([[-np.inf], thetas, [np.inf]])
    rows = []
    for th in thetas:
        cost, err, done, lvl = (np.zeros(len(Wd)) for _ in range(4))
        done = done.astype(bool)
        for i, L in enumerate(levels):
            acc = ~done & ((Wd[f"{score}_{L}"].to_numpy(float) <= th) | (i == len(levels) - 1))
            cost[acc], err[acc], lvl[acc] = Wd[f"cost_{L}"].to_numpy(float)[acc], Wd[f"err_{L}"].to_numpy(float)[acc], i
            done |= acc
        rows.append(dict(theta=th, cost=cost.mean(), viol=float((err > tau).mean()), mean_level=lvl.mean()))
    return pd.DataFrame(rows)


def fixed_points(Wd, levels, tau):
    import pandas as pd
    return pd.DataFrame([dict(level=L, cost=Wd[f"cost_{L}"].mean(), viol=float((Wd[f"err_{L}"] > tau).mean()))
                         for L in levels])


def oracle_frontier(Wd, levels, tau, n_points: int = 101):
    """Cheapest cost for each violation budget when the true error is known per slice."""
    import pandas as pd
    costs = np.stack([Wd[f"cost_{L}"].to_numpy(float) for L in levels], 1)
    ok = np.stack([Wd[f"err_{L}"].to_numpy(float) <= tau for L in levels], 1)
    first = np.where(ok.any(1), ok.argmax(1), len(levels) - 1)
    c_ok = costs[np.arange(len(Wd)), first]
    unfixable = ~ok.any(1)
    saving = np.where(unfixable, 0.0, c_ok - costs[:, 0])
    order = np.argsort(-saving)
    rows = []
    n = len(Wd)
    for v in np.linspace(0, 1, n_points):
        m = int(np.floor(v * n)) - int(unfixable.sum())
        c = c_ok.copy()
        if m > 0:
            sel = [i for i in order if not unfixable[i]][:m]
            c[sel] = costs[sel, 0]
        rows.append(dict(viol_budget=v, cost=c.mean(), viol=min(1.0, (max(m, 0) + unfixable.sum()) / n)))
    return pd.DataFrame(rows)


def cost_at(curve, v: float) -> float:
    ok = curve[curve.viol <= v + 1e-12]
    return float(ok.cost.min()) if len(ok) else float("nan")


def timeshare_cost(fixed, v: float) -> float:
    """Cheapest mixture of fixed-level protocols (assigned at random) with violation <= v."""
    pts = fixed[["cost", "viol"]].to_numpy(float)
    best = np.inf
    for i in range(len(pts)):
        for j in range(len(pts)):
            (c1, v1), (c2, v2) = pts[i], pts[j]
            if v1 <= v:
                best = min(best, c1)
            if v1 > v >= v2 and v1 != v2:
                lam = (v1 - v) / (v1 - v2)
                best = min(best, (1 - lam) * c1 + lam * c2)
    return float(best)


def savings_captured(Wd, levels, tau, v: float, score: str = "pred") -> dict:
    cur = cascade_curve(Wd, levels, tau, score)
    fixed, orc = fixed_points(Wd, levels, tau), oracle_frontier(Wd, levels, tau)
    cf, cu, co = timeshare_cost(fixed, v), cost_at(cur, v), cost_at(orc, v)
    frac = (cf - cu) / (cf - co) if np.isfinite(cf) and np.isfinite(cu) and cf - co > 1e-9 else float("nan")
    return dict(viol_target=v, cost_fixed=cf, cost_score=cu, cost_oracle=co, captured=frac)


def ltt_cascade(Wd, units, levels, tau, alpha: float, delta: float = 0.1, score: str = "pred",
                n_splits: int = 200, frac: float = 0.5, seed: int = 0, method: str = "ltt"):
    """Threshold for the cascade chosen on calibration units, evaluated on held-out units
    (repeated random unit splits). method='ltt': Learn-then-Test fixed sequence, Clopper-Pearson
    (1 - delta) upper bound on the violation rate <= alpha (high-probability guarantee; needs
    many calibration slices). method='crc': conformal risk control, (n * viol + 1) / (n + 1) <= alpha
    (guarantee in expectation; the violation is monotone in theta up to per-slice non-monotone errors)."""
    import pandas as pd
    from scipy.stats import beta
    rng = np.random.default_rng(seed)
    U = np.unique(units)
    rows = []
    for s in range(n_splits):
        cal = set(rng.choice(U, max(1, int(round(frac * len(U)))), replace=False))
        mc = np.array([u in cal for u in units])
        cc = cascade_curve(Wd[mc], levels, tau, score).sort_values("theta")
        n = mc.sum()
        th = -np.inf
        for _, r in cc.iterrows():
            k = int(round(r.viol * n))
            if method == "crc":
                ucb = (k + 1) / (n + 1)
            else:
                ucb = 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))
            if ucb > alpha:
                break
            th = r.theta
        te = cascade_curve(Wd[~mc], levels, tau, score, thetas=[th]).iloc[0]
        rows.append(dict(split=s, theta=th, test_viol=te.viol, test_cost=te.cost))
    return pd.DataFrame(rows)


def allocation_curve(base_err, gain, score, cost_base: float, cost_extra: float, n_points: int = 21):
    """Give the extra averages to the top-q fraction of slices ranked by score (higher = more
    expected gain). Returns mean cost and mean error for score, oracle (true gain) and random."""
    import pandas as pd
    base_err, gain, score = (np.asarray(a, float) for a in (base_err, gain, score))
    rows = []
    for q in np.linspace(0, 1, n_points):
        m = int(round(q * len(gain)))
        for name, s in (("score", score), ("oracle", gain)):
            sel = np.argsort(-s)[:m]
            e = base_err.copy()
            e[sel] -= gain[sel]
            rows.append(dict(policy=name, q=q, cost=cost_base + q * cost_extra, err=e.mean()))
        rows.append(dict(policy="random", q=q, cost=cost_base + q * cost_extra, err=base_err.mean() - q * gain.mean()))
    return pd.DataFrame(rows)


def quantile_from_hist(h, edges, q: float) -> float:
    c = np.cumsum(h) / max(np.sum(h), 1)
    i = int(np.searchsorted(c, q))
    return float(edges[min(i + 1, len(edges) - 1)])


def coverage_from_hist(h, edges, thr_log10: float) -> float:
    centers = 0.5 * (edges[1:] + edges[:-1])
    h = np.asarray(h, float)
    return float(h[centers <= thr_log10].sum() / max(h.sum(), 1))


def column_stats(rec) -> dict:
    """Per phase-encode column: calibration (sum of actual error / predicted variance) and rank
    agreement, separately on unmeasured and measured columns. 'dist' ranks columns by closeness to
    the k-space centre, the trivial baseline (the analytic spectrum ranks columns the same way)."""
    err, ens, ana = (np.asarray(rec[k], float) for k in ("col_err", "col_ens", "col_ana"))
    meas, acq = np.asarray(rec["col_meas"], bool), np.asarray(rec["col_acq"], bool)
    W = len(err)
    dist = np.abs(np.arange(W) - W // 2)
    u = acq & ~meas
    out = {}
    if u.sum() >= 5:
        out.update(ratio_ens_unmeas=err[u].sum() / max(ens[u].sum(), 1e-30),
                   ratio_ana_unmeas=err[u].sum() / max(ana[u].sum(), 1e-30),
                   rho_ens_unmeas=_spearman(ens[u], err[u]), rho_dist_unmeas=_spearman(-dist[u], err[u]))
    if meas.sum() >= 5:
        out.update(ratio_ens_meas=err[meas].sum() / max(ens[meas].sum(), 1e-30),
                   ratio_ana_meas=err[meas].sum() / max(ana[meas].sum(), 1e-30))
    return out


def maha_fit(X: np.ndarray):
    X = np.asarray(X, float)
    C = np.cov(X.T) + 1e-6 * np.eye(X.shape[1])
    return X.mean(0), np.linalg.inv(C)


def maha(X: np.ndarray, mu: np.ndarray, Ci: np.ndarray) -> np.ndarray:
    d = np.asarray(X, float) - mu
    return np.einsum("ij,jk,ik->i", d, Ci, d)


def verdict(value: float, promising: float, weak: float, higher_is_better: bool = True) -> str:
    if value is None or not np.isfinite(value):
        return "NOT RUN"
    v, p, w = (value, promising, weak) if higher_is_better else (-value, -promising, -weak)
    return "PROMISING" if v >= p else "WEAK" if v >= w else "NO SIGNAL"
