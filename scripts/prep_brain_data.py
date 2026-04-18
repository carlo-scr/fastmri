"""Prepare brain multicoil data: select volumes and optionally crop.

Copies a subset of brain multicoil HDF5 files to a working directory,
verifying they contain valid k-space data at the expected resolution.

Usage:
    python scripts/prep_brain_data.py \
        --input_dir data/brain_multicoil_val \
        --output_dir data/brain_val \
        --num_volumes 10 \
        --target_resolution 384 320
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prepare brain multicoil data")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_volumes", type=int, default=10)
    parser.add_argument("--target_resolution", type=int, nargs=2, default=None,
                        help="Filter volumes whose k-space H,W >= target [H W]")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(input_dir.glob("*.h5"))
    if not h5_files:
        print(f"ERROR: No HDF5 files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(h5_files)} volumes in {input_dir}")

    copied = 0
    skipped = 0
    for f in h5_files:
        if copied >= args.num_volumes:
            break

        with h5py.File(f, "r") as hf:
            # Prefer reconstruction_rss for resolution check (val/test sets)
            if "reconstruction_rss" in hf:
                shape = hf["reconstruction_rss"].shape
                num_slices, H, W = shape
                num_coils = hf["kspace"].shape[1] if len(hf["kspace"].shape) == 4 else 1
            elif len(hf["kspace"].shape) == 4:
                shape = hf["kspace"].shape
                num_slices, num_coils, H, W = shape
            else:
                print(f"  SKIP {f.name}: unexpected format")
                skipped += 1
                continue

            if args.target_resolution:
                tH, tW = args.target_resolution
                if H < tH or W < tW:
                    print(f"  SKIP {f.name}: {H}×{W} < {tH}×{tW}")
                    skipped += 1
                    continue

        dest = output_dir / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
        print(f"  [{copied+1}/{args.num_volumes}] {f.name} — {num_slices} slices, {num_coils} coils, {H}×{W}")
        copied += 1

    print(f"\nDone: {copied} volumes copied to {output_dir} ({skipped} skipped)")


if __name__ == "__main__":
    main()
