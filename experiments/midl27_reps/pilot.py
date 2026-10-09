"""Pilot runners: data diagnostics (P0), training data (P1), reconstruction evaluation (P3).

All outputs are JSON/JSONL so the notebook can resume after a disconnect.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE), str(HERE.parent / "midl27")):
    if p not in sys.path:
        sys.path.insert(0, p)

import fdc  # noqa: E402
import m4raw as M  # noqa: E402
from train_reps import TwoRegime, truncate_schedule  # noqa: E402


def subject_kspace(split: str, subject: str, contrast: str, cat: dict, listing: list, raw_dir: str,
                   n_slices: int, fit_gain: bool = False):
    """Fetch, register and whiten one subject/contrast. Returns (Kw, acq, slices, reg_info, psi)."""
    paths = M.fetch(split, cat[subject][contrast], raw_dir, listing)
    K, idx = M.load_reps(paths, n_slices)
    acq = M.acquired_columns(K)
    Kr, info = M.register_reps(K, fit_gain=fit_gain)
    psi = M.psi_from_reps(Kr, acq)
    return M.whiten(Kr, psi), acq, idx, info, psi


# ---------------------------------------------------------------------------
# P0: data diagnostics and kill criteria
# ---------------------------------------------------------------------------

def diagnostics(split, subjects, cat, listing, raw_dir, n_slices=4, T=50, sigma_max=10.0) -> list[dict]:
    from src.samplers.schedules import edm_sigma_schedule
    sig = edm_sigma_schedule(T, sigma_max=sigma_max).numpy()
    rows = []
    for s in subjects:
        for c in M.CONTRASTS:
            if c not in cat[s]:
                continue
            Kw, acq, idx, info, psi = subject_kspace(split, s, c, cat, listing, raw_dir, n_slices)
            dg = np.real(np.diag(psi))
            corr = np.abs(psi / np.sqrt(np.outer(dg, dg)) - np.eye(len(dg))).max()
            for z in range(Kw.shape[1]):
                meas = M.prepare(Kw[:, z], acq, R=4, mask_seed=0, device="cpu")
                m1 = M.corrected_metrics(meas["x1_full"], meas)
                ri = [i for i in info if i["slice"] == z]
                x1 = meas["x1_full"]
                snr = float(np.median(np.abs(x1)[meas["region"]]) / np.sqrt(np.mean(meas["nv1"][meas["region"]])))
                row = dict(subject=s, contrast=c, slice=idx[z], n_reps=Kw.shape[0], acq_cols=int(acq.sum()),
                           psi_max_corr=float(corr), snr_median=snr, floor=meas["floor"],
                           frac_steps_below_floor=float((sig < meas["floor"]).mean()),
                           physio_frac=m1["physio_frac"], ena_full1=m1["ena"],
                           max_shift_px=float(max([abs(i["dy"]) + abs(i["dx"]) for i in ri] or [0])),
                           max_phase_deg=float(max([abs(i["phase_deg"]) for i in ri] or [0])))
                for e in M.estimator_check(Kw[:, z], acq):
                    row[f"check_{e['estimator']}_ena"] = e["ena"]
                rows.append(row)
            print(f"[P0] {s} {c}: floor {np.mean([r['floor'] for r in rows[-len(idx):]]):.4f}, "
                  f"physio {np.mean([r['physio_frac'] for r in rows[-len(idx):]]):.3f}")
    return rows


# ---------------------------------------------------------------------------
# P1: training slices
# ---------------------------------------------------------------------------

def build_training(split, subjects, cat, listing, raw_dir, n_slices=8, cache=None) -> list[dict]:
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return list(z["slices"])
    out = []
    for i, s in enumerate(subjects):
        for c in M.CONTRASTS:
            if c in cat[s]:
                Kw, acq, idx, _, _ = subject_kspace(split, s, c, cat, listing, raw_dir, n_slices)
                for d in M.training_slices(Kw, acq, seed=i):
                    d["x"] = d["x"].astype(np.complex64)
                    out.append(dict(d, subject=s, contrast=c))
        print(f"[P1] {i + 1}/{len(subjects)} subjects, {len(out)} slices")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache + ".tmp.npz", slices=np.array(out, dtype=object))
        os.replace(cache + ".tmp.npz", cache)
    return out


# ---------------------------------------------------------------------------
# P3: reconstruction evaluation (resumable JSONL)
# ---------------------------------------------------------------------------

def _done(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                keys.add((r["subject"], r["contrast"], r["slice"], r["R"], r["method"], r.get("kappa"), r.get("T")))
            except (json.JSONDecodeError, KeyError):
                pass
    return keys


def l1_wavelet(meas: dict, lamda: float) -> np.ndarray:
    import sigpy.mri as mr
    y = meas["y"].cpu().numpy()
    w = meas["op"].mask.cpu().numpy()
    return mr.app.L1WaveletRecon(y, meas["sens"], lamda, weights=w, max_iter=100, show_pbar=False).run()


def evaluate(split, subjects, cat, listing, raw_dir, out_path, *, methods: dict, Rs=(2, 4), T=50,
             kappas=(1.0,), lamdas=(0.003,), n_slices=4, contrasts=M.CONTRASTS, sigma_max=10.0,
             device="cuda", seed=0, sync=None) -> str:
    """methods: name -> dict(kind='zf'|'l1'|'sampler', denoiser=..., stop_at_floor=bool,
    cov='iso'|'spectral'). Every record is one (subject, contrast, slice, R, method, kappa)."""
    from src.samplers.schedules import edm_sigma_schedule
    done = _done(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base_spec = fdc.standard_methods()
    t0, n_new = time.time(), 0
    for s in subjects:
        for c in contrasts:
            if c not in cat[s]:
                continue
            Kw, acq, idx, _, _ = subject_kspace(split, s, c, cat, listing, raw_dir, n_slices)
            for z in range(Kw.shape[1]):
                for R in Rs:
                    meas = None
                    for name, m in methods.items():
                        grid = kappas if m["kind"] == "sampler" else lamdas if m["kind"] == "l1" else (None,)
                        for kap in grid:
                            key = (s, c, idx[z], R, name, kap, T if m["kind"] == "sampler" else None)
                            if key in done:
                                continue
                            if meas is None:
                                meas = M.prepare(Kw[:, z], acq, R=R, mask_seed=zlib.crc32(f'{s}|{c}|{idx[z]}'.encode()) % 2**31,
                                                 device=device)
                            t1 = time.time()
                            if m["kind"] == "zf":
                                xh = meas["op"].AH(meas["y"]).cpu().numpy()
                            elif m["kind"] == "full1":
                                xh = meas["x1_full"]
                            elif m["kind"] == "l1":
                                xh = l1_wavelet(meas, kap)
                            else:
                                spec = copy.deepcopy(base_spec["ikdc_fakgd" if m.get("cov") == "spectral" else "ikdc_iso"])
                                spec.kappa = kap
                                sig = edm_sigma_schedule(T, sigma_max=sigma_max).to(meas["y"].device)
                                den = m["denoiser"]
                                if m.get("stop_at_floor"):
                                    sig = truncate_schedule(sig, meas["floor"])
                                elif m.get("two_regime"):
                                    den = TwoRegime(m["denoiser"], m["below"], meas["floor"])
                                res = fdc.run_sampler(spec, denoiser=den, op=meas["op"], y=meas["y"], s2=meas["s2"],
                                                      sigmas=sig, center_fraction=meas["cf"], spectrum=meas["spectrum"],
                                                      data_var=meas["data_var"], seed=seed)
                                xh = res["recon"].cpu().numpy()
                            rec = dict(subject=s, contrast=c, slice=idx[z], R=R, acc=meas["acc"], method=name,
                                       kappa=kap, T=T if m["kind"] == "sampler" else None, seed=seed,
                                       runtime_s=time.time() - t1, floor=meas["floor"], n_held=meas["n_held"],
                                       **M.corrected_metrics(xh, meas), cr_mse=M.common_ref_mse(xh, meas),
                                       **M.averaging_baselines(meas))
                            with open(out_path, "a") as f:
                                f.write(json.dumps(rec) + "\n")
                            n_new += 1
            print(f"[P3] {s} {c} done ({n_new} new records, {(time.time() - t0) / 60:.1f} min)")
            if sync:
                sync()
    return out_path


# ---------------------------------------------------------------------------
# Selection and gate analysis
# ---------------------------------------------------------------------------

def load_records(path: str):
    import pandas as pd
    return pd.DataFrame([json.loads(l) for l in open(path)]) if os.path.exists(path) else pd.DataFrame()


def select_params(path: str) -> dict:
    """Best kappa (samplers) / lambda (L1) per (method, R) by mean corrected PSNR on the tuning split."""
    df = load_records(path)
    best = {}
    for (m, R), g in df.groupby(["method", "R"]):
        means = g.groupby("kappa", dropna=False)["psnr_corr"].mean()
        best[f"{m}|R{R}"] = None if means.index.isna().all() else float(means.idxmax())
    return best


def _boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def gate_report(path: str, ft_methods=("ft_stop", "ft_two"), base="zeroshot", min_gain_db=1.0) -> dict:
    """P3 gate. Unit = subject x contrast (slices averaged). GO: best post-trained variant is
    >= min_gain_db over zero-shot with a 95% CI above 0 at both R, and beats the fully sampled
    2-average at R=2 (common reference, contrasts with >= 3 repetitions) in >= 50% of units."""
    df = load_records(path)
    unit = ["subject", "contrast"]
    out = {"per_R": {}}
    for R, g in df.groupby("R"):
        agg = g.groupby(unit + ["method"]).agg(psnr=("psnr_corr", "mean"), ena=("ena", "median"),
                                               cr=("cr_mse", "mean"), cr1=("cr_full1", "mean"),
                                               cr2=("cr_full2", "mean")).reset_index()
        piv = agg.pivot_table(index=unit, columns="method", values="psnr")
        rows = {}
        for m in sorted(agg.method.unique()):
            a = agg[agg.method == m]
            rows[m] = dict(psnr=float(a.psnr.mean()), ena_median=float(a.ena.median()),
                           beats_1avg=float((a.cr < a.cr1).mean()) if a.cr.notna().any() else None,
                           beats_2avg=float((a.cr < a.cr2).mean()) if a.cr.notna().any() else None)
            if m != base and base in piv and m in piv:
                d = (piv[m] - piv[base]).dropna()
                rows[m].update(delta_vs_base=float(d.mean()), ci=_boot(d), n_units=int(len(d)),
                               n_positive=int((d > 0).sum()))
        out["per_R"][int(R)] = rows
    cands = [m for m in ft_methods if all(m in out["per_R"].get(R, {}) for R in out["per_R"])]
    def ok(m):
        return all(out["per_R"][R][m].get("delta_vs_base", -1) >= min_gain_db and out["per_R"][R][m]["ci"][0] > 0
                   for R in out["per_R"])
    best = max(cands, key=lambda m: np.mean([out["per_R"][R][m].get("delta_vs_base", -9) for R in out["per_R"]]), default=None)
    r2 = out["per_R"].get(2, {}).get(best, {}) if best else {}
    out["best_ft"] = best
    out["GO"] = bool(best and ok(best) and (r2.get("beats_2avg") or 0) >= 0.5)
    out["STRONG"] = bool(out["GO"] and (r2.get("beats_2avg") or 0) >= 0.8)
    return out
