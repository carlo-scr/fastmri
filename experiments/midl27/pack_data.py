"""Pack a small, deterministic subset of fastMRI multi-coil volumes for Colab.

Keeps only the central ``--n_slices`` slices of each volume (k-space plus
attributes), so a 24-volume brain cohort is a few GB instead of ~9 GB.

    python experiments/midl27/pack_data.py \
        --src data/multicoil_val --acquisition AXFLAIR \
        --n_volumes 28 --n_slices 8 --out midl27_brain.tar

Upload the tar to MyDrive/fastmri_artifacts/ and point the notebook at it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import tarfile
import tempfile

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--acquisition", default="AXFLAIR")
    ap.add_argument("--n_volumes", type=int, default=28)
    ap.add_argument("--n_slices", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.src, "*.h5")))
    if a.acquisition:
        files = [f for f in files if f"_{a.acquisition}_" in os.path.basename(f)]
    order = np.random.RandomState(a.seed).permutation(len(files))[: a.n_volumes]
    chosen = [files[i] for i in sorted(order)]
    with tempfile.TemporaryDirectory() as tmp:
        cohort = os.path.join(tmp, "midl27_cohort")
        os.makedirs(cohort)
        for f in chosen:
            with h5py.File(f, "r") as h:
                nz = h["kspace"].shape[0]
                lo = max(0, nz // 2 - a.n_slices // 2)
                idx = list(range(lo, min(nz, lo + a.n_slices)))
                with h5py.File(os.path.join(cohort, os.path.basename(f)), "w") as o:
                    o.create_dataset("kspace", data=h["kspace"][idx])
                    for k, v in h.attrs.items():
                        o.attrs[k] = v
                    o.attrs["midl27_source_slices"] = json.dumps(idx)
            print("packed", os.path.basename(f))
        with open(os.path.join(cohort, "PACKING.json"), "w") as fp:
            json.dump({"src": a.src, "acquisition": a.acquisition, "seed": a.seed,
                       "n_slices": a.n_slices, "files": [os.path.basename(f) for f in chosen]}, fp, indent=1)
        with tarfile.open(a.out, "w") as t:
            t.add(cohort, arcname="midl27_cohort")
    print("wrote", a.out, f"({os.path.getsize(a.out) / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
