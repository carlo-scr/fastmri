# MIDL 2027 data acquisition (runs in Colab, not on the Mac)

`data_acquisition_colab.ipynb` is self-contained. Upload it to Colab (File → Upload notebook)
or paste its cells, set the flags in cell 0, then Run all. A CPU runtime is enough. Raw
downloads stay on the runtime disk. Only packed tars and checkpoints go to
`MyDrive/fastmri_artifacts/`, the folder `../midl27_colab.ipynb` already reads.
Edit `build_notebook.py` and re-run it to regenerate the notebook.

| Step | Source | Output on Drive | Paper |
|---|---|---|---|
| A | [fastMRI+](https://github.com/microsoft/fastmri-plus) `Annotations/*.csv` | `fastmri_plus/` | Tab. 6 |
| B | InverseBench release `MRI-knee.pt`, fastMRI `knee_leaderboard_state_dict.pt` | `checkpoints/` | Tab. 3 knee |
| C | [M4Raw, Zenodo 8056074](https://zenodo.org/records/8056074): test 5.0 GB, val 3.0 GB | `m4raw_test.tar`, `m4raw_calib.tar` | Tab. 5, E3 |
| D | fastMRI knee multi-coil val (NYU licence link → `KNEE_URL`), streamed with early stop | `midl27_knee.tar` | Tab. 3, Tab. 6 |
| E | — | `DATA_MANIFEST.json` (sizes + SHA-256) | Tab. 1 frozen manifest |

Tar layout matches `../pack_data.py`: one folder of h5 files (`kspace` = central slices,
original attrs, `midl27_source_slices`) plus `PACKING.json`. K-space is gzip-compressed inside
the h5. h5py decompresses it transparently, so `run.load_volume` reads it unchanged.

## Open issues these data raise (not solved here)
- **Knee prior format.** `MRI-knee.pt` is InverseBench's `EDMPrecond`/DhariwalUNet at
  **320×320** MVUE, not an Ambient-DPS pickle at 384×320. `run.load_denoiser` and the crop
  target need a knee branch.
- **M4Raw noise estimation.** Our Ψ estimate uses signal-free readout rows of an oversampled
  FOV. M4Raw k-space appears to be 256×256. Step C.2 prints whether empty margins exist. If
  they don't, Ψ needs another source, such as image background or the repetitions.
- **Knee cohort is pathology-enriched** when `KNEE_SELECT='boxes'`. Report this in the setup section.
- **Brain AXT2 has no fastMRI+ review** in the local val files (0/211 reviewed). For the
  local AXFLAIR cohort, 12/28 volumes have boxes, on any slice.
