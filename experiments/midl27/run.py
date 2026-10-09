"""Experiment runner for the MIDL 2027 study (resumable, JSONL output).

Typical use (from the notebook, or the command line):

    python experiments/midl27/run.py --config experiments/midl27/configs/g1_test.yaml

Every record is one (volume, slice, R, T, method, seed) reconstruction.
Re-running skips records already present in the output file.
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import fdc  # noqa: E402
from fdc import MethodSpec  # noqa: E402

CENTER_FRACTION = {4: 0.08, 8: 0.04}


# ---------------------------------------------------------------------------
# Manifest: deterministic calibration / test split by volume
# ---------------------------------------------------------------------------

def sha256(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def make_manifest(data_dir: str, out_path: str, n_calib: int, n_test: int,
                  acquisition: str | None = None, seed: int = 2027,
                  hash_files: bool = True) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    if acquisition:
        files = [f for f in files if f"_{acquisition}_" in os.path.basename(f)]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(files))
    if n_calib + n_test > len(files):
        raise ValueError(f"need {n_calib + n_test} volumes, found {len(files)} in {data_dir}")
    calib = [files[i] for i in order[:n_calib]]
    test = [files[i] for i in order[n_calib:n_calib + n_test]]
    man = {
        "data_dir": data_dir, "seed": seed, "acquisition": acquisition,
        "calib": [os.path.basename(f) for f in calib],
        "test": [os.path.basename(f) for f in test],
        "sha256": {os.path.basename(f): sha256(f) for f in calib + test} if hash_files else {},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(man, f, indent=1)
    return man


# ---------------------------------------------------------------------------
# Data: prewhitened, cropped multi-coil k-space + references
# ---------------------------------------------------------------------------

@dataclass
class SliceData:
    volume: str
    slice_idx: int
    y_full: torch.Tensor      # (Nc, H, W) prewhitened full k-space (raw units)
    sens: torch.Tensor        # (Nc, H, W) ESPIRiT maps (shared by recon and target)
    x_mvue: torch.Tensor      # (H, W) complex MVUE target, sum_c S_c^* x_c / sum_c |S_c|^2
    ref_mag: torch.Tensor     # (H, W) |x_mvue| -- the evaluation target (as Ambient-DPS)


def load_volume(path: str, n_slices: int, target=(384, 320), noise_scale: float = 0.0,
                noise_seed: int = 0, calib_width: int = 24) -> tuple[list[SliceData], dict]:
    """Central ``n_slices`` of a volume, prewhitened with a per-volume Psi.

    ``noise_scale`` > 0 adds white noise (identity covariance after whitening,
    i.e. Psi-correlated in raw coordinates) with variance noise_scale^2 x the
    native noise -- used only for the controlled SNR sweep.

    Coil maps: ESPIRiT on the central ``calib_width`` fully sampled columns,
    shared by reconstruction and the MVUE target. This is the convention of
    the prior's training data (Ambient-DPS) and of Jalal et al. (2021).
    """
    with h5py.File(path, "r") as f:
        Nz = f["kspace"].shape[0]
        lo = max(0, Nz // 2 - n_slices // 2)
        idx = list(range(lo, min(Nz, lo + n_slices)))
        k = f["kspace"][idx]                                  # (n, Nc, H0, W0)
    psi = fdc.estimate_psi(k)
    Wm = torch.from_numpy(fdc.whitening_matrix(psi)).to(torch.complex64)
    out = []
    gen = torch.Generator().manual_seed(noise_seed)
    for j, sl in enumerate(idx):
        kw = fdc.apply_whitening(torch.from_numpy(k[j]).to(torch.complex64), Wm)
        if noise_scale > 0:
            n = torch.complex(torch.randn(kw.shape, generator=gen), torch.randn(kw.shape, generator=gen))
            kw = kw + noise_scale * n / np.sqrt(2.0)
        coil = fdc.center_crop(fdc.ifft2c(kw), *target)
        y_full = fdc.fft2c(coil)
        sens = fdc.espirit_maps(y_full, calib_width=calib_width)
        den = sens.abs().square().sum(0)
        x_mvue = torch.where(den > 0, (sens.conj() * coil).sum(0) / den.clamp(min=1e-12),
                             torch.zeros_like(coil[0]))
        out.append(SliceData(os.path.basename(path), sl, y_full, sens, x_mvue, x_mvue.abs()))
    meta = {"psi_trace": float(np.real(np.trace(psi))), "psi_cond": float(np.linalg.cond(psi)),
            "n_coils": int(k.shape[1]), "native_shape": list(k.shape[-2:]),
            "noise_var_after_whitening": float(1.0 + noise_scale ** 2)}
    return out, meta


def prepare_measurement(sd: SliceData, R: int, mask_type: str, mask_seed: int, device,
                        noise_var: float = 1.0) -> dict:
    """Mask, normalise (ACS-only q99 RSS scale), sensitivities, spectrum."""
    Nc, H, W = sd.y_full.shape
    cf = CENTER_FRACTION[R]
    mfun = fdc.equispaced_mask if mask_type == "equispaced" else fdc.random_mask
    m1 = mfun(W, R, cf, mask_seed)
    y_full = sd.y_full.to(device)
    # zero-padded (never acquired) columns are excluded from the mask
    acquired = (y_full.abs().sum(dim=(0, 1)) > 0).float().cpu()
    m1 = m1 * acquired
    mask = m1[None, :].expand(H, W).contiguous().to(device)
    y = mask * y_full
    # scale from acquired ACS only
    aw = max(2, int(round(cf * W)))
    acs = torch.zeros_like(y)
    acs[..., W // 2 - aw // 2: W // 2 + aw // 2] = y[..., W // 2 - aw // 2: W // 2 + aw // 2]
    lr = fdc.ifft2c(acs).abs().square().sum(0).sqrt()
    scale = float(torch.quantile(lr.flatten().float(), 0.99))
    y, y_full = y / scale, y_full / scale
    s2 = noise_var / scale ** 2
    sens = sd.sens.to(device)
    op = fdc.SenseOp(mask, sens)
    spec = fdc.estimate_spectrum(y, mask, s2)
    rho = fdc.radius_grid(H, W, device=device)
    data_var = float(spec(rho).mean())
    x_ref = sd.x_mvue.to(device) / scale                     # complex target (diagnostics)
    return {"op": op, "y": y, "y_full": y_full, "s2": s2, "scale": scale, "spectrum": spec,
            "data_var": data_var, "x_ref": x_ref, "mask_cols": int(m1.sum()), "cf": cf}


def oracle_spectrum(meas: dict) -> fdc.Spectrum:
    acquired = (meas["y_full"].abs().sum(dim=(0, 1)) > 0).float()
    full = acquired[None, :].expand_as(meas["op"].mask).contiguous()
    return fdc.estimate_spectrum(meas["y_full"], full, meas["s2"], mode="bins")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_denoiser(ckpt_dir: str, device: str):
    from src.models.edm_loader import load_edm_model, EDMDenoiser
    net = load_edm_model(ckpt_dir, method="auto", device=device)
    for p in net.parameters():
        p.requires_grad_(False)
    return EDMDenoiser(net, device=device)


# ---------------------------------------------------------------------------
# One reconstruction
# ---------------------------------------------------------------------------

def reconstruct(spec: MethodSpec, meas: dict, denoiser, T: int, sigma_max: float,
                seed: int, diagnostics: bool = False) -> dict:
    from src.samplers.schedules import edm_sigma_schedule
    sigmas = edm_sigma_schedule(T, sigma_max=sigma_max).to(meas["y"].device)
    spectrum = meas["spectrum"]
    if spec.cov == "spectral" and spec.spec_mode in ("bins", "oracle"):
        spectrum = oracle_spectrum(meas) if spec.spec_mode == "oracle" else copy.copy(spectrum)
        spectrum.mode = "bins"
    nfe0 = denoiser.nfe
    t0 = time.time()
    if spec.host == "fakgd_v1":
        res = fdc.run_fakgd_v1(spec, denoiser=denoiser, op=meas["op"], y=meas["y"], s2=meas["s2"],
                               sigmas=sigmas, center_fraction=meas["cf"], seed=seed)
    else:
        res = fdc.run_sampler(spec, denoiser=denoiser, op=meas["op"], y=meas["y"], s2=meas["s2"],
                              sigmas=sigmas, center_fraction=meas["cf"], spectrum=spectrum,
                              data_var=meas["data_var"], seed=seed, x_ref=meas["x_ref"],
                              diagnostics=diagnostics)
    if meas["y"].is_cuda:
        torch.cuda.synchronize()
    res["runtime_s"] = time.time() - t0
    res["nfe"] = denoiser.nfe - nfe0
    return res


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def _key(rec) -> tuple:
    return (rec["volume"], rec["slice"], rec["R"], rec["T"], rec["method"], rec["seed"],
            rec.get("snr_tag", "native"))


def load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    done.add(_key(json.loads(line)))
                except json.JSONDecodeError:
                    pass
    return done


def resolve_methods(cfg: dict) -> list[MethodSpec]:
    """cfg['methods']: list of names from standard_methods() or dicts
    {name, base, **overrides}; tuned values in cfg['tuned'] (json path) are
    applied to iso baselines per (R, T) at run time."""
    std = fdc.standard_methods()
    out = []
    for m in cfg["methods"]:
        if isinstance(m, str):
            out.append(copy.deepcopy(std[m]))
        else:
            m = dict(m)
            base = copy.deepcopy(std[m.pop("base")]) if "base" in m else MethodSpec(m["name"])
            for k, v in m.items():
                setattr(base, k, v)
            out.append(base)
    return out


def apply_tuned(spec: MethodSpec, tuned: dict, R: int, T: int) -> MethodSpec:
    key = f"{spec.name}|R{R}|T{T}"
    alt = f"{spec.name}|R{R}|T*"
    params = tuned.get(key) or tuned.get(alt)
    if not params:
        return spec
    spec = copy.deepcopy(spec)
    for k, v in params.items():
        setattr(spec, k, v)
    return spec


def run_grid(cfg: dict) -> str:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    man = json.load(open(cfg["manifest"]))
    vols = man[cfg.get("split", "test")][: cfg.get("n_volumes", 10 ** 9)]
    out_path = cfg["output"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = load_done(out_path)
    tuned = json.load(open(cfg["tuned"])) if cfg.get("tuned") and os.path.exists(cfg["tuned"]) else {}
    methods = resolve_methods(cfg)
    denoiser = load_denoiser(cfg["checkpoint"], device)
    snr_scales = cfg.get("noise_scales", [0.0])
    n_total = len(vols) * cfg["n_slices"] * len(cfg["R"]) * len(cfg["T"]) * len(methods) \
        * len(cfg.get("seeds", [0])) * len(snr_scales)
    print(f"[run_grid] {len(vols)} volumes, up to {n_total} records -> {out_path}")
    n_new = 0
    t_start = time.time()
    with open(out_path, "a") as fout:
        for vi, vname in enumerate(vols):
            for ns in snr_scales:
                snr_tag = "native" if ns == 0 else f"noise_x{ns:g}"
                slices, vmeta = load_volume(os.path.join(man["data_dir"], vname), cfg["n_slices"],
                                            noise_scale=ns, noise_seed=vi)
                drange = float(max(float(s.ref_mag.max()) for s in slices))
                for sd in slices:
                    vseed = int(hashlib.md5(f"{vname}:{sd.slice_idx}".encode()).hexdigest()[:8], 16)
                    for R in cfg["R"]:
                        meas = None
                        for T in cfg["T"]:
                            for spec0 in methods:
                                spec = apply_tuned(spec0, tuned, R, T)
                                for seed in cfg.get("seeds", [0]):
                                    rec_key = (vname, sd.slice_idx, R, T, spec.name, seed, snr_tag)
                                    if rec_key in done:
                                        continue
                                    if meas is None:
                                        meas = prepare_measurement(sd, R, cfg.get("mask", "equispaced"),
                                                                   vseed % (2 ** 31), device,
                                                                   noise_var=1.0 + ns ** 2)
                                    res = reconstruct(spec, meas, denoiser, T, cfg.get("sigma_max", 10.0),
                                                      seed=vseed + 1000 * seed,
                                                      diagnostics=cfg.get("diagnostics", False))
                                    rec_mag = res["recon"].abs() * meas["scale"]
                                    met = fdc.slice_metrics(rec_mag.cpu(), sd.ref_mag, drange,
                                                            fg=(sd.ref_mag > 0.05 * drange))
                                    rec = {"volume": vname, "slice": sd.slice_idx, "R": R, "T": T,
                                           "method": spec.name, "seed": seed, "snr_tag": snr_tag,
                                           "split": cfg.get("split", "test"), "spec": spec.to_dict(),
                                           "metrics": met, "runtime_s": res["runtime_s"], "nfe": res["nfe"],
                                           "data_range": drange, "mask_cols": meas["mask_cols"],
                                           "scale": meas["scale"], "s2": meas["s2"],
                                           "spectrum_coef": list(meas["spectrum"].coef),
                                           "volume_meta": vmeta}
                                    if "m_step_rel_change" in res:
                                        rec["m_step_rel_change"] = res["m_step_rel_change"]
                                    if "diagnostics" in res:
                                        rec["diagnostics"] = res["diagnostics"]
                                    if device == "cuda":
                                        rec["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                                    fout.write(json.dumps(rec) + "\n")
                                    fout.flush()
                                    done.add(rec_key)
                                    n_new += 1
                                    if n_new % 10 == 0:
                                        el = time.time() - t_start
                                        print(f"  {n_new} new records, {el / 60:.1f} min, "
                                              f"last {spec.name} R{R} T{T}: {met['psnr']:.2f} dB "
                                              f"({res['runtime_s']:.1f}s)")
            print(f"[run_grid] volume {vi + 1}/{len(vols)} done")
    print(f"[run_grid] wrote {n_new} new records in {(time.time() - t_start) / 60:.1f} min")
    return out_path


# ---------------------------------------------------------------------------
# Tuning of isotropic baselines (equal grid per sampler, calibration split)
# ---------------------------------------------------------------------------

DEFAULT_GRIDS = {
    # isotropic baselines: tuned per (R, T) on calibration volumes
    "ikdc_iso": {"kappa": [0.1, 0.3, 1.0, 3.0, 10.0]},
    "pigdm_iso": {"guidance": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "dds_iso": {"n_cg": [3, 5, 10], "eta": [0.0, 0.15, 0.5]},
    "diffpir_iso": {"kappa": [0.3, 1.0, 3.0], "eta": [0.3, 0.6, 1.0]},
    "dps": {"guidance": [0.1, 0.3, 1.0, 3.0, 10.0]},
}

# FA-KGD rows: ONE configuration per host, chosen on calibration at T=50 and
# then used unchanged for every (R, T) cell (paper Sec. 4, 'fair tuning').
FAKGD_GRIDS = {
    "ikdc_fakgd": {"kappa": [0.3, 1.0, 3.0]},
    "pigdm_fakgd": {"guidance": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "dds_fakgd": {"n_cg": [3, 5, 10], "eta": [0.0, 0.15, 0.5]},
    "diffpir_fakgd": {"kappa": [0.3, 1.0, 3.0], "eta": [0.3, 0.6, 1.0]},
}


def tuning_configs(grids: dict) -> list[dict]:
    """Expand {method: {param: [values]}} into method dicts named method@k=v,..."""
    import itertools
    out = []
    for name, g in grids.items():
        keys = sorted(g)
        for vals in itertools.product(*[g[k] for k in keys]):
            tag = ",".join(f"{k}={v}" for k, v in zip(keys, vals))
            out.append({"name": f"{name}@{tag}", "base": name, **dict(zip(keys, vals))})
    return out


def select_tuned(results_path: str, out_path: str, fakgd_T: int = 50) -> dict:
    """Best config per (method, R, T) by volume-mean PSNR on calibration.
    FA-KGD rows get one config per (host, R), chosen at T = fakgd_T."""
    import analysis
    df = analysis.load(results_path)
    df = df[df.method.str.contains("@")]
    df["base"] = df.method.str.split("@").str[0]
    vol = analysis.volume_table(df, extra_keys=["base"])
    best = {}

    def parse(m):
        out = {}
        for kv in m.split("@")[1].split(","):
            k, v = kv.split("=")
            out[k] = int(float(v)) if k == "n_cg" else float(v)
        return out

    for (base, R, T), g in vol.groupby(["base", "R", "T"]):
        m = g.groupby("method").psnr.mean().idxmax()
        if base.endswith("_fakgd"):
            if T == fakgd_T:
                best[f"{base}|R{R}|T*"] = parse(m)
        else:
            best[f"{base}|R{R}|T{T}"] = parse(m)
    with open(out_path, "w") as f:
        json.dump(best, f, indent=1)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    import yaml
    run_grid(yaml.safe_load(open(a.config)))
