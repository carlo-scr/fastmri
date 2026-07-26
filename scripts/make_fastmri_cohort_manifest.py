#!/usr/bin/env python3
"""Create an immutable, explicitly selected fastMRI cohort manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py


BRAIN_ACQUISITIONS = {"AXT1", "AXT1PRE", "AXT1POST", "AXT2", "AXFLAIR"}
KNEE_ACQUISITIONS = {"CORPD_FBK", "CORPDFS_FBK"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument(
        "--filenames",
        nargs="+",
        required=True,
        help="Exact basenames selected before hashing; no directory auto-selection.",
    )
    parser.add_argument("--dataset_revision", required=True)
    parser.add_argument("--anatomy", choices=["brain", "knee"], required=True)
    parser.add_argument("--expected_count", type=int, default=20)
    parser.add_argument("--minimum_height", type=int, default=384)
    parser.add_argument("--minimum_width", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset_revision.strip().lower() in {"", "unspecified"}:
        raise ValueError("dataset_revision must identify the exact snapshot")
    if args.expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if len(args.filenames) != args.expected_count:
        raise ValueError(
            f"expected exactly {args.expected_count} explicit filenames, "
            f"received {len(args.filenames)}"
        )
    if len(set(args.filenames)) != len(args.filenames):
        raise ValueError("filenames must be unique")
    if any(Path(name).name != name for name in args.filenames):
        raise ValueError("filenames must be plain basenames")
    allowed = (
        BRAIN_ACQUISITIONS
        if args.anatomy == "brain"
        else KNEE_ACQUISITIONS
    )

    volumes = []
    for filename in args.filenames:
        path = args.data_path / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            if "kspace" not in handle or "reconstruction_rss" not in handle:
                raise ValueError(
                    f"{filename} lacks kspace or reconstruction_rss"
                )
            kspace_shape = list(handle["kspace"].shape)
            rss_shape = list(handle["reconstruction_rss"].shape)
            acquisition_value = handle.attrs.get("acquisition", "unspecified")
            acquisition = (
                acquisition_value.decode("utf-8")
                if isinstance(acquisition_value, bytes)
                else str(acquisition_value)
            )
        if acquisition not in allowed:
            raise ValueError(
                f"{filename}: acquisition {acquisition!r} is not {args.anatomy}"
            )
        if (
            kspace_shape[-2] < args.minimum_height
            or kspace_shape[-1] < args.minimum_width
            or rss_shape[-2] < args.minimum_height
            or rss_shape[-1] < args.minimum_width
        ):
            raise ValueError(
                f"{filename}: native support kspace={kspace_shape[-2:]}, "
                f"rss={rss_shape[-2:]} is smaller than "
                f"{(args.minimum_height, args.minimum_width)}"
            )
        volumes.append({
            "filename": filename,
            "sha256": sha256_file(path),
            "acquisition": acquisition,
            "kspace_shape": kspace_shape,
            "rss_shape": rss_shape,
        })
    hashes = [row["sha256"] for row in volumes]
    if len(set(hashes)) != len(hashes):
        raise ValueError(
            "selected filenames contain duplicate volume file contents"
        )

    manifest = {
        "schema_version": 1,
        "dataset": "fastMRI",
        "dataset_revision": args.dataset_revision,
        "split": "validation",
        "anatomy": args.anatomy,
        "selection_policy": "explicit_filenames_selected_before_hashing",
        "volumes": volumes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"wrote {args.output}")
    print(f"manifest SHA256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
