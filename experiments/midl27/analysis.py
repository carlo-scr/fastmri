"""Volume-level statistics, gates, figures and LaTeX table rows.

Unit of analysis = volume (paper Sec. 4, 'Statistics'). Slice numbers are
descriptive only. PSNR per volume is computed from stacked sufficient
statistics (sum of squared errors over all evaluated slices, volume max as
data range), not by averaging slice PSNRs.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd


def load(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.pop("metrics")
            r.pop("diagnostics", None)
            r.pop("spec", None)
            r.pop("volume_meta", None)
            rows.append({**r, **m})
    return pd.DataFrame(rows)


def volume_table(df: pd.DataFrame, extra_keys=()) -> pd.DataFrame:
    keys = ["volume", "R", "T", "method", "snr_tag", "seed", *extra_keys]
    g = df.groupby(keys)
    vol = g.agg(sse=("sse", "sum"), numel=("numel", "sum"), ref_sq=("ref_sq", "sum"),
                drange=("data_range", "max"), ssim=("ssim", "mean"), hfen=("hfen", "mean"),
                runtime_s=("runtime_s", "mean"), nfe=("nfe", "mean"), n_slices=("slice", "count")).reset_index()
    vol["psnr"] = 10 * np.log10(vol.drange ** 2 / (vol.sse / vol.numel))
    vol["nmse"] = vol.sse / vol.ref_sq
    # average over seeds last
    keys_ns = [k for k in keys if k != "seed"]
    return vol.groupby(keys_ns).mean(numeric_only=True).reset_index()


def bootstrap_ci(d: np.ndarray, n: int = 10000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    if len(d) < 2:
        return (float("nan"), float("nan"))
    bs = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def paired(vol: pd.DataFrame, a: str, b: str, metric: str = "psnr", **where) -> dict:
    """Paired volume-level difference a - b (e.g. a='pigdm_fakgd', b='pigdm_iso')."""
    from scipy.stats import wilcoxon
    sub = vol
    for k, v in where.items():
        sub = sub[sub[k] == v]
    A = sub[sub.method == a].set_index("volume")[metric]
    B = sub[sub.method == b].set_index("volume")[metric]
    common = A.index.intersection(B.index)
    d = (A[common] - B[common]).to_numpy()
    out = {"a": a, "b": b, "metric": metric, **where, "n_volumes": int(len(d)),
           "mean_a": float(A[common].mean()) if len(d) else float("nan"),
           "mean_b": float(B[common].mean()) if len(d) else float("nan"),
           "delta": float(d.mean()) if len(d) else float("nan"),
           "ci": bootstrap_ci(d), "n_positive": int((d > 0).sum())}
    if len(d) >= 5 and np.any(d != 0):
        out["p_wilcoxon_one_sided"] = float(wilcoxon(d, alternative="greater").pvalue)
    else:
        out["p_wilcoxon_one_sided"] = float("nan")
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj.tolist()


# ---------------------------------------------------------------------------
# Gates (revision_plan.tex, Sec. 7)
# ---------------------------------------------------------------------------

def gate_g1(vol: pd.DataFrame, host: str = "ikdc", min_delta: float = 0.2,
            Ts=(50, 100), Rs=(4, 8)) -> pd.DataFrame:
    """G1: on real noise, +FA-KGD beats the *tuned* isotropic baseline by
    >= min_delta dB with a 95% CI excluding 0, for T >= 50."""
    rows = []
    for R in Rs:
        for T in Ts:
            r = paired(vol, f"{host}_fakgd", f"{host}_iso", R=R, T=T, snr_tag="native")
            r["pass"] = bool(r["delta"] >= min_delta and r["ci"][0] > 0)
            rows.append(r)
    return pd.DataFrame(rows)


def mstep_report(df: pd.DataFrame) -> pd.DataFrame:
    """C3: is the CS231N M-step inert? Compares fakgd_v1 off/clamp/full."""
    sub = df[df.method.str.startswith("fakgd_v1")]
    vol = volume_table(sub)
    rows = []
    for (R, T), g in vol.groupby(["R", "T"]):
        for m in ("fakgd_v1_clamp", "fakgd_v1_full"):
            if m in set(g.method):
                r = paired(vol, m, "fakgd_v1_off", R=R, T=T)
                rel = sub[(sub.method == m) & (sub.R == R) & (sub["T"] == T)].get("m_step_rel_change")
                r["mean_rel_change_sigma2"] = float(rel.mean()) if rel is not None else float("nan")
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX helpers (paste rows into MIDL2027/paper/sections/results.tex)
# ---------------------------------------------------------------------------

def fmt(x, nd=2, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return r"\xx"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("+", "$+$").replace("-", "$-$") if signed else s


def fmt_ci(ci, nd=2):
    return f"[{fmt(ci[0], nd, True)},\\,{fmt(ci[1], nd, True)}]"


def steps_table_rows(vol: pd.DataFrame, R: int = 8, hosts=("ikdc", "pigdm"),
                     Ts=(10, 20, 50, 100, 200)) -> str:
    lines = []
    for T in Ts:
        cells = [str(T)]
        for h in hosts:
            r = paired(vol, f"{h}_fakgd", f"{h}_iso", R=R, T=T, snr_tag="native")
            cells += [fmt(r["mean_b"]), fmt(r["mean_a"]), f"{fmt(r['delta'], 2, True)} {fmt_ci(r['ci'])}"]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def main_table_rows(vol: pd.DataFrame, T: int = 50, hosts=("ikdc", "pigdm", "dds", "diffpir"),
                    cells=(("brain", 4), ("brain", 8), ("knee", 4), ("knee", 8))) -> str:
    """Expects an 'anatomy' column (add it when concatenating cohorts)."""
    lines = []
    for h in hosts:
        for suffix, label in (("iso", "isotropic"), ("fakgd", r"+\method")):
            row = [rf"\multirow{{2}}{{*}}{{{h}}}" if suffix == "iso" else "", label]
            for anat, R in cells:
                s = vol[(vol.method == f"{h}_{suffix}") & (vol.R == R) & (vol["T"] == T)]
                if "anatomy" in vol:
                    s = s[s.anatomy == anat]
                row += [fmt(s.psnr.mean()), fmt(s.ssim.mean(), 3)]
            lines.append(" & ".join(row) + r" \\")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_steps(vol: pd.DataFrame, R: int, hosts=("ikdc", "pigdm", "dds", "diffpir"), ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    for i, h in enumerate(hosts):
        for suffix, ls in (("iso", "-"), ("fakgd", "--")):
            s = vol[(vol.method == f"{h}_{suffix}") & (vol.R == R)]
            if s.empty:
                continue
            g = s.groupby("T").psnr
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med.values, ls, color=f"C{i}", label=f"{h} {suffix}")
            ax.fill_between(med.index, q1.values, q3.values, color=f"C{i}", alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("reverse steps T")
    ax.set_ylabel("volume PSNR (dB)")
    ax.legend(fontsize=7, ncol=2)
    return ax


def load_diagnostics(path: str, method: str, R: int, T: int) -> dict:
    band, calib = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["method"] == method and r["R"] == R and r["T"] == T and "diagnostics" in r:
                band.append(r["diagnostics"]["band_nmse"])
                calib.append(r["diagnostics"]["calib_ratio"])
    return {"band_nmse": np.array(band), "calib_ratio": np.array(calib, dtype=float)}
