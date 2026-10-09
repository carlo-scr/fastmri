"""M4Raw repetition pipeline for the MIDL 2027 repetition-supervision pilot.

Data access, registration, noise model, coil combination, the noise-corrected
evaluation and the training examples. Plan: MIDL2027/proposal_repetitions/.

Facts this module relies on (checked on val subject 2022061203, 2026-10-09):
  * Zenodo zips are *stored* (no compression): any single h5 (~12 MB) can be
    range-fetched, no zip download needed.
  * kspace (18 slices, 4 coils, 256, 256) complex64; FOV 240 mm, no readout
    oversampling; phase-encode columns ~30..225 acquired, the rest zero-padded.
  * Repetitions differ by a global phase (-16..-64 deg) and sub-pixel shifts
    (<= 0.33 px); after correction the difference is noise-only.
  * Coil noise correlation up to 0.25 -> prewhiten with Psi estimated from
    repetition differences (no noise prescans, no signal-free readout margin).

Conventions: centred orthonormal FFTs (fdc.fft2c). After whitening each coil's
k-space has unit complex noise variance on acquired samples. Noise levels
called ``sigma`` are per *real* channel (EDM convention), i.e. complex var / 2.
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import struct
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "midl27"))
import fdc  # noqa: E402

ZENODO = "https://zenodo.org/api/records/8056074/files/{}/content"
SPLIT_ZIP = {"train": "M4RawV1.5_multicoil_train.zip",
             "val": "M4RawV1.5_multicoil_val.zip",
             "test": "M4Raw_multicoil_test.zip"}
NAME_RE = re.compile(r"(\d+)_(T1|T2|FLAIR)(\d+)\.h5$")
CONTRASTS = ("T1", "T2", "FLAIR")


# ---------------------------------------------------------------------------
# Remote access: list a stored zip and range-fetch single members
# ---------------------------------------------------------------------------

def _range(url: str, a: int, b: int, tries: int = 5) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except OSError:
            if i == tries - 1:
                raise
    raise IOError(f"short read {url} [{a},{b}]")


def zip_listing(split: str, cache: str | None = None) -> list[dict]:
    """Central directory of a split's zip (ZIP64-aware), via HTTP range requests."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    url = ZENODO.format(SPLIT_ZIP[split])
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req) as r:
        n = int(r.headers["Content-Range"].split("/")[1])
    tail = _range(url, n - 65558, n - 1)
    i = tail.rfind(b"PK\x05\x06")
    total = struct.unpack("<H", tail[i + 10:i + 12])[0]
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if 0xFFFFFFFF in (cd_size, cd_off) or total == 0xFFFF:
        j = tail.rfind(b"PK\x06\x07")
        z64 = struct.unpack("<Q", tail[j + 8:j + 16])[0]
        total, cd_size, cd_off = struct.unpack("<QQQ", _range(url, z64, z64 + 55)[32:56])
    cd = _range(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        csz, usz = struct.unpack("<II", cd[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", cd[p + 28:p + 34])
        loff, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + fl].decode()
        extra, q = cd[p + 46 + fl:p + 46 + fl + el], 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hsz // 8), extra[q + 4:q + 4 + hsz]))
                if usz == 0xFFFFFFFF: usz = vals.pop(0)
                if csz == 0xFFFFFFFF: csz = vals.pop(0)
                if loff == 0xFFFFFFFF: loff = vals.pop(0)
            q += 4 + hsz
        base = name.split("/")[-1]
        if base.endswith(".h5") and not base.startswith("._"):
            out.append(dict(name=base, method=meth, csize=csz, usize=usz, offset=loff))
        p += 46 + fl + el + cl
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        json.dump(out, open(cache, "w"))
    return out


def fetch(split: str, names: list[str], out_dir: str, listing: list[dict]) -> list[str]:
    """Range-fetch stored zip members to out_dir (skips files already complete)."""
    url = ZENODO.format(SPLIT_ZIP[split])
    by = {e["name"]: e for e in listing}
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for n in names:
        e, dst = by[n], os.path.join(out_dir, n)
        if e["method"] != 0:
            raise ValueError(f"{n} is compressed in the zip; range fetch needs stored members")
        if not (os.path.exists(dst) and os.path.getsize(dst) == e["csize"]):
            h = _range(url, e["offset"], e["offset"] + 29)
            fl, el = struct.unpack("<HH", h[26:30])
            start = e["offset"] + 30 + fl + el
            data = _range(url, start, start + e["csize"] - 1)
            with open(dst + ".part", "wb") as f:
                f.write(data)
            os.replace(dst + ".part", dst)
        paths.append(dst)
    return paths


def catalog(listing: list[dict]) -> dict:
    """{subject: {contrast: [names sorted by repetition]}}"""
    cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in listing:
        m = NAME_RE.search(e["name"])
        if m:
            cat[m[1]][m[2]].append((int(m[3]), e["name"]))
    return {s: {c: [n for _, n in sorted(v)] for c, v in d.items()} for s, d in sorted(cat.items())}


def pick_subjects(cat: dict, n: int | None, seed: int, exclude=()) -> list[str]:
    subs = [s for s in sorted(cat) if s not in set(exclude)]
    if n is None or n >= len(subs):
        return subs
    return sorted(random.Random(seed).sample(subs, n))


# ---------------------------------------------------------------------------
# Loading, registration, noise covariance, whitening
# ---------------------------------------------------------------------------

def central_slices(nz: int, n: int) -> list[int]:
    lo = max(0, nz // 2 - n // 2)
    return list(range(lo, min(nz, lo + n)))


def load_reps(paths: list[str], n_slices: int) -> tuple[np.ndarray, list[int]]:
    """(rep, slice, coil, H, W) complex64 for the central n_slices."""
    with h5py.File(paths[0], "r") as h:
        idx = central_slices(h["kspace"].shape[0], n_slices)
    K = []
    for p in paths:
        with h5py.File(p, "r") as h:
            K.append(h["kspace"][idx])
    return np.stack(K).astype(np.complex64), idx


def acquired_columns(K: np.ndarray) -> np.ndarray:
    """(W,) bool: phase-encode columns that carry data (K: (..., H, W))."""
    return np.abs(K).reshape(-1, K.shape[-1]).sum(0) > 0


def _ifft(k: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(k, axes=(-2, -1)), axes=(-2, -1), norm="ortho"),
                           axes=(-2, -1))


def _kgrid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(H) - H // 2)[:, None] / H, (np.arange(W) - W // 2)[None, :] / W


def shift_kspace(k: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Exact sub-pixel image shift by +d via a linear phase on centred k-space."""
    ky, kx = _kgrid(*k.shape[-2:])
    return k * np.exp(-2j * np.pi * (ky * dy + kx * dx)).astype(np.complex64)


def fit_shift_phase(k0: np.ndarray, kr: np.ndarray) -> tuple[float, float, float]:
    """(dy, dx, phase) with kr ~ e^{i phase} shift(k0, d), from the coil-summed complex
    cross-power spectrum (coarse: upsampled phase correlation; fine: Nelder-Mead)."""
    from scipy.optimize import minimize
    from skimage.registration import phase_cross_correlation
    ky, kx = _kgrid(*k0.shape[-2:])
    C = (kr * k0.conj()).sum(0)
    m0, mr = (np.sqrt((np.abs(_ifft(k)) ** 2).sum(0)) for k in (k0, kr))
    s, _, _ = phase_cross_correlation(m0, mr, upsample_factor=20, normalization=None)
    f = lambda p: -np.abs(np.sum(C * np.exp(2j * np.pi * (ky * p[0] + kx * p[1]))))
    r = minimize(f, -np.asarray(s, float), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-10))
    dy, dx = (float(v) for v in r.x)
    ph = float(np.angle(np.sum(C * np.exp(2j * np.pi * (ky * dy + kx * dx)))))
    return dy, dx, ph


def register_reps(K: np.ndarray, fit_gain: bool = False) -> tuple[np.ndarray, list[dict]]:
    """Align every repetition to repetition 0, per slice: sub-pixel shift and global phase
    (and, optionally, amplitude from the high-SNR k-space centre).

    K: (rep, slice, coil, H, W). Returns registered K and per (rep, slice) info.
    """
    Kr = K.copy()
    info = []
    H, W = K.shape[-2:]
    ky, kx = _kgrid(H, W)
    centre = (ky ** 2 + kx ** 2) < 0.05 ** 2
    for z in range(K.shape[1]):
        for r in range(1, K.shape[0]):
            dy, dx, ph = fit_shift_phase(K[0, z], K[r, z])
            kr = shift_kspace(K[r, z], -dy, -dx) * np.complex64(np.exp(-1j * ph))
            gain = 1.0
            if fit_gain:
                gain = float(np.real(np.sum(kr[:, centre].conj() * K[0, z][:, centre]))
                             / np.sum(np.abs(K[0, z][:, centre]) ** 2))
                kr = kr / np.float32(gain)
            Kr[r, z] = kr
            info.append(dict(rep=r, slice=z, dy=dy, dx=dx, phase_deg=float(np.degrees(ph)), gain=gain))
    return Kr, info


def outer_region(H: int, W: int, acq: np.ndarray, r: float = 0.30) -> np.ndarray:
    fy = (np.arange(H) - H // 2)[:, None] / H
    fx = (np.arange(W) - W // 2)[None, :] / W
    return ((fy ** 2 + fx ** 2) > r ** 2) & acq[None, :]


def psi_from_reps(Kr: np.ndarray, acq: np.ndarray) -> np.ndarray:
    """Coil noise covariance from differences of registered repetitions (outer, acquired k-space)."""
    R, Z, C, H, W = Kr.shape
    sel = outer_region(H, W, acq)
    X = []
    for r in range(1, R):
        X.append(((Kr[r] - Kr[r - 1]) / np.sqrt(2))[:, :, sel].transpose(1, 0, 2).reshape(C, -1))
    X = np.concatenate(X, axis=1)
    return (X @ X.conj().T / X.shape[1]).astype(np.complex128)


def whiten(Kr: np.ndarray, psi: np.ndarray) -> np.ndarray:
    Wm = fdc.whitening_matrix(psi).astype(np.complex64)
    return np.einsum("ij,...jhw->...ihw", Wm, Kr).astype(np.complex64)


# ---------------------------------------------------------------------------
# Coil combination, normalisation, noise maps
# ---------------------------------------------------------------------------

ESPIRIT_CROP = 0.0   # the fastMRI default (0.95) zeroes ~10% of the head at 0.3 T SNR and leaves
                     # 35% of the coil signal unexplained; crop=0 leaves 10% (val 2022061203, T1)


def coil_maps(k_full: np.ndarray, calib_width: int) -> np.ndarray:
    """ESPIRiT maps (C, H, W) from the central calib_width columns of one slice."""
    return fdc.espirit_maps(torch.from_numpy(k_full), calib_width=calib_width, crop=ESPIRIT_CROP).numpy()


def mvue(k: np.ndarray, sens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Complex MVUE image and den = sum_c |S_c|^2 (0 outside the map support)."""
    coil = _ifft(k)
    den = (np.abs(sens) ** 2).sum(0)
    x = np.where(den > 1e-6, (sens.conj() * coil).sum(0) / np.maximum(den, 1e-6), 0)
    return x.astype(np.complex64), den.astype(np.float32)


def acs_scale(k: np.ndarray, acs_cols: int) -> float:
    """99th percentile of the RSS of the low-resolution ACS image (run.py convention)."""
    W = k.shape[-1]
    a = np.zeros_like(k)
    a[..., W // 2 - acs_cols // 2: W // 2 + acs_cols // 2] = k[..., W // 2 - acs_cols // 2: W // 2 + acs_cols // 2]
    return float(np.quantile(np.sqrt((np.abs(_ifft(a)) ** 2).sum(0)), 0.99))


def mvue_noise_var(den: np.ndarray, acq_frac: float) -> np.ndarray:
    """Complex noise variance per pixel of a whitened single-repetition MVUE."""
    return np.where(den > 1e-6, acq_frac / np.maximum(den, 1e-6), 0).astype(np.float32)


def support_region(x: np.ndarray, den: np.ndarray, frac: float = 0.08, blur: float = 2.0) -> np.ndarray:
    """Evaluation region: smoothed |x| above frac x its 99th percentile, inside the map support."""
    from scipy.ndimage import gaussian_filter, binary_fill_holes
    m = gaussian_filter(np.abs(x), blur)
    reg = (m > frac * np.quantile(m, 0.99)) & (den > 1e-6)
    return binary_fill_holes(reg)


# ---------------------------------------------------------------------------
# Measurement for one (subject, contrast, slice): reconstruct from repetition 0
# ---------------------------------------------------------------------------

CENTER_FRACTION = {2: 0.08, 3: 0.08, 4: 0.08, 6: 0.06, 8: 0.04}


def prepare(Kw: np.ndarray, acq: np.ndarray, R: int, mask_seed: int, device,
            calib_width: int = 20, recon_rep: int = 0) -> dict:
    """Kw: (rep, coil, H, W) whitened, registered k-space of one slice.

    Everything the reconstruction may use comes from repetition ``recon_rep``
    only (maps, scale, measurements), so the held-out repetitions' noise is
    independent of the reconstruction and the corrected error is unbiased.
    """
    nrep, C, H, W = Kw.shape
    k1 = Kw[recon_rep]
    sens = coil_maps(k1, calib_width)
    cf = CENTER_FRACTION[R]
    m1 = fdc.equispaced_mask(W, R, cf, mask_seed).numpy() * acq
    scale = acs_scale(k1, int(round(cf * W)))
    y_full = torch.from_numpy(k1 / scale)
    mask = torch.from_numpy(np.broadcast_to(m1[None], (H, W)).copy()).float()
    y = mask * y_full
    acq_frac = float(acq.mean())
    x1, den = mvue(k1 / scale, sens)
    held = [r for r in range(nrep) if r != recon_rep]
    xs = [mvue(Kw[r] / scale, sens)[0] for r in range(nrep)]
    ref = np.mean([xs[r] for r in held], axis=0)
    nv1 = mvue_noise_var(den, acq_frac) / scale ** 2            # one repetition, thermal complex var
    region = support_region(x1, den)
    v_rep = repetition_variance(xs, region)                    # thermal + physiological, region mean
    sens_t = torch.from_numpy(sens).to(device)
    mask_d, y_d = mask.to(device), y.to(device)
    s2 = 1.0 / scale ** 2
    spec = fdc.estimate_spectrum(y_d, mask_d, s2)
    rho = fdc.radius_grid(H, W, device=device)
    return dict(op=fdc.SenseOp(mask_d, sens_t), y=y_d, s2=s2, scale=scale, cf=cf, spectrum=spec,
                data_var=float(spec(rho).mean()), x1_full=x1, ref=ref.astype(np.complex64),
                ref_var=nv1 / len(held), nv1=nv1, region=region, v_rep=v_rep,
                ref_last=xs[-1] if nrep >= 3 else None,
                x2avg=0.5 * (xs[recon_rep] + xs[held[0]]) if nrep >= 3 else None,
                floor=float(np.sqrt(np.quantile(nv1[den > 1e-6], 0.99) / 2)),
                acc=float(acq.sum() / max(m1.sum(), 1)), n_held=len(held), sens=sens, mask_cols=m1)


# ---------------------------------------------------------------------------
# Noise-corrected evaluation
# ---------------------------------------------------------------------------

def repetition_variance(xs: list[np.ndarray], region: np.ndarray) -> float:
    """Region-mean variance of one repetition around the expected image, from all pairs:
    E|x_r - x_s|^2 / 2 = thermal + physiological (inter-repetition) variance."""
    pairs = [(r, s) for r in range(len(xs)) for s in range(r + 1, len(xs))]
    return float(np.mean([np.mean(np.abs(xs[r] - xs[s])[region] ** 2) / 2 for r, s in pairs]))


def corrected_metrics(xhat: np.ndarray, meas: dict) -> dict:
    """Unbiased estimate of the error against the expected image (mean over repetitions).

    The held-out average deviates from the expected image by thermal noise and by
    inter-repetition (physiological) variation; both are independent of xhat, which
    uses repetition 0 only. With v_rep the per-repetition total variance,
        E|xhat - ref|^2 = |xhat - x|^2 + v_rep / n_held.
    ENA = v_rep / corrected MSE: 1 = one fully sampled average, K = K averages.
    ena_thermal uses thermal noise only (ignores physiological variation; optimistic).
    """
    reg = meas["region"]
    raw = float(np.mean(np.abs(xhat - meas["ref"])[reg] ** 2))
    v_rep = meas["v_rep"]
    corr = raw - v_rep / meas["n_held"]
    nv = float(np.mean(meas["nv1"][reg]))
    corr_th = raw - nv / meas["n_held"]
    return dict(mse_raw=raw, mse_corr=corr, psnr_corr=float(10 * np.log10(1.0 / max(corr, 1e-12))),
                ena=float(v_rep / corr) if corr > 0 else float("inf"),
                ena_thermal=float(nv / corr_th) if corr_th > 0 else float("inf"),
                v_rep=v_rep, noise_var_1avg=nv, physio_frac=float(v_rep / nv - 1))


def common_ref_mse(xhat: np.ndarray, meas: dict) -> float | None:
    """Raw MSE against the last repetition only. Every candidate scored this way shares the
    same additive noise term, so differences between candidates are unbiased: this is the
    bias-free test of "beats a 1-average / 2-average fully sampled scan" (needs >= 3 reps)."""
    if meas.get("ref_last") is None:
        return None
    return float(np.mean(np.abs(xhat - meas["ref_last"])[meas["region"]] ** 2))


def averaging_baselines(meas: dict) -> dict:
    """Common-reference MSE of the fully sampled 1-average and 2-average scans."""
    if meas.get("ref_last") is None:
        return {}
    return dict(cr_full1=common_ref_mse(meas["x1_full"], meas), cr_full2=common_ref_mse(meas["x2avg"], meas))


def estimator_check(Kw: np.ndarray, acq: np.ndarray, calib_width: int = 20) -> list[dict]:
    """Sanity check on slices with >= 3 repetitions: a fully sampled single repetition must
    score ENA ~ 1, and the average of two repetitions ENA ~ 2."""
    out = []
    if Kw.shape[0] < 3:
        return out
    meas = prepare(Kw, acq, R=4, mask_seed=0, device="cpu", calib_width=calib_width)
    out.append(dict(estimator="full_1avg", **corrected_metrics(meas["x1_full"], meas)))
    # average of reps 0 and 1 against rep 2..: rebuild reference from reps >= 2 only
    sens, scale = meas["sens"], meas["scale"]
    x01 = 0.5 * (mvue(Kw[0] / scale, sens)[0] + mvue(Kw[1] / scale, sens)[0])
    ref = np.mean([mvue(Kw[r] / scale, sens)[0] for r in range(2, Kw.shape[0])], axis=0)
    m2 = dict(meas, ref=ref, ref_var=meas["nv1"] / (Kw.shape[0] - 2), n_held=Kw.shape[0] - 2)
    out.append(dict(estimator="full_2avg", **corrected_metrics(x01, m2)))
    return out


# ---------------------------------------------------------------------------
# Training examples for repetition score matching
# ---------------------------------------------------------------------------

def training_slices(Kw: np.ndarray, acq: np.ndarray, calib_width: int = 24, seed: int = 0) -> list[dict]:
    """Per slice: complex images of every repetition (noise made white by filling the
    never-acquired columns with matched noise), and the per-real-channel noise std map.

    Kw: (rep, slice, coil, H, W) whitened and registered.
    """
    rng = np.random.default_rng(seed)
    R, Z, C, H, W = Kw.shape
    out = []
    for z in range(Z):
        kavg = Kw[:, z].mean(0)
        sens = coil_maps(kavg, calib_width)
        scale = acs_scale(kavg, int(round(0.08 * W)))
        xs = []
        for r in range(R):
            k = Kw[r, z].copy()
            fill = (rng.standard_normal(k.shape) + 1j * rng.standard_normal(k.shape)) / np.sqrt(2)
            k[..., ~acq] = fill[..., ~acq]
            x, den = mvue(k / scale, sens)
            xs.append(x)
        sig = np.where(den > 1e-6, np.sqrt(1.0 / np.maximum(den, 1e-6) / 2) / scale, 0).astype(np.float32)
        out.append(dict(x=np.stack(xs), sig=sig, floor=float(np.quantile(sig[den > 1e-6], 0.99))))
    return out
