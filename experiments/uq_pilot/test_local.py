"""Execute uq_pilot_colab.ipynb end to end on the CPU with a tiny configuration.

Skips the Colab-only cells (tag 'setup') and replaces them with a local setup, then runs every
other cell (runs, analysis, scoreboard) in one namespace. Numbers are meaningless (T = 2); the
point is that every code path and analysis cell executes.

    python experiments/uq_pilot/test_local.py --fastmri data/multicoil_val --m4raw_raw /tmp/m4raw_raw \
        --out /tmp/uq_smoke --subject 2022061203
"""
import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastmri", required=True, help="folder with fastMRI brain multi-coil h5 files")
    ap.add_argument("--acquisition", default="AXFLAIR")
    ap.add_argument("--m4raw_raw", required=True, help="cache folder for range-fetched M4Raw files")
    ap.add_argument("--subject", default="2022061203")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", action="store_true", help="reuse records from a previous run (analysis only)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(max(1, os.cpu_count() - 2))
    os.chdir(ROOT)
    for p in (str(ROOT), str(HERE), str(ROOT / "experiments" / "midl27_reps")):
        sys.path.insert(0, p)

    if not a.keep:
        shutil.rmtree(a.out, ignore_errors=True)
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    fdir = out / "fastmri"
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir()
    for f in sorted(glob.glob(os.path.join(a.fastmri, f"*_{a.acquisition}_*.h5")))[:2]:
        os.symlink(os.path.abspath(f), fdir / os.path.basename(f))

    nb = json.loads((HERE / "uq_pilot_colab.ipynb").read_text())
    ns = {"display": lambda x: print(x.to_string() if hasattr(x, "to_string") else x)}
    setup_done = False
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if "setup" in c.get("metadata", {}).get("tags", []):
            if not setup_done:
                exec(LOCAL_SETUP, ns, ns)
                ns.update(OUT=str(out / "results"), RAW=a.m4raw_raw, MAPS=str(out / "maps"), FASTMRI_DIR=str(fdir))
                exec(LOCAL_DATA, ns, ns)
                cat = ns["cat_val"]
                ns["EXCLUDE"] = {s for s in cat if s != a.subject}
                setup_done = True
            continue
        print(f"\n===== cell {i}: {src.splitlines()[0][:90]}")
        exec(compile(src, f"cell{i}", "exec"), ns, ns)
        if src.lstrip().startswith("# 0.1"):
            ns.update(DEVICE="cpu", T=2, K=3, N_FOLDS=1, N_FASTMRI_VOLS=2, FASTMRI_SLICES=(6, 9),
                      N_M4RAW_SUBJ=1, M4RAW_CONTRASTS=("T1", "FLAIR"), N_G=2, BUDGET_MIN=10 ** 6)
    print("\nOK: every non-setup cell executed")


LOCAL_SETUP = r"""
import sys, json, time, shutil, tarfile, copy, os
import numpy as np, torch, pandas as pd
import uq, m4raw as M
from src.models.edm_loader import load_edm_model
T_START = time.time()
def sync(verbose=False):
    pass
base_net = load_edm_model('checkpoints/edm/supervised_R=1', method='auto', device='cpu').eval().requires_grad_(False)
den = uq.make_denoiser(base_net)
FT_PATH = None
"""

LOCAL_DATA = r"""
lst_val = uq.retry(lambda: M.zip_listing('val', cache=f'{RAW}/listing_val.json'))
cat_val = M.catalog(lst_val)
"""

if __name__ == "__main__":
    main()
