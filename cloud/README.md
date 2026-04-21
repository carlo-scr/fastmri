# Cloud data hosting (Google Cloud Storage)

Scripts for mirroring the fastMRI brain multi-coil validation set into a
private GCS bucket and mounting it on a training VM. The repo's brain
evaluation notebook (`notebooks/brain/02_brain_edm_evaluation.ipynb`)
expects these files; storing them on GCS avoids ~94 GB of local disk per
collaborator and keeps the data available to everyone running experiments.

## Data-use-agreement constraint

The fastMRI dataset is shared under NYU's data use agreement, which
**prohibits redistribution**. The scripts here enforce:

- Uniform bucket-level access.
- Public-access prevention (`enforced`).
- Read/write limited to named project members.

Do not relax these settings. Do not make the bucket public or share
signed URLs outside the team.

## One-time setup

```bash
# From any machine with gcloud installed and authenticated as a project admin.
# Defaults are wired to the project we're using; override with env vars if needed.
./cloud/setup_bucket.sh
```

This creates the bucket, applies the access policy, binds IAM roles for
the two Stanford collaborators, and installs a lifecycle rule that
deletes every object 90 days after its custom time (see "Retention"
below).

## Pulling data from NYU into the bucket

NYU serves the dataset as time-limited presigned S3 URLs. The upload
script streams a URL straight into the bucket — nothing lands on local
disk — so it works from Cloud Shell, a small Compute Engine VM, or a
laptop with decent bandwidth.

```bash
# Get fresh presigned URLs from https://fastmri.med.nyu.edu (they expire).
export FASTMRI_URL='<paste the full presigned URL for brain_multicoil_val_batch_0>'
./cloud/upload_from_nyu.sh "$FASTMRI_URL" brain_multicoil_val_batch_0.tar.xz
```

Repeat with the `batch_1` and `batch_2` URLs if you need the full
validation set (~276 GB total). The repo's brain evaluation uses a
handful of volumes from `batch_0`, so that alone (~94 GB) is usually
enough.

## Using the data: run compute where the data lives

The bucket is in `us-central1`. Two cost/performance realities follow
from that:

- A Compute Engine VM in the **same region** reads the bucket for
  **free** (intra-region traffic).
- Any machine in a **different region or outside Google Cloud** (your
  laptop, Colab if not in `us-central1`, another cloud) pays **egress
  fees** — roughly **$0.12/GB**. Pulling the full 94 GB validation
  batch to a laptop costs ~$11 each time.

So the recommended pattern is: create a `us-central1` VM, run the
experiments there, write results back to the bucket. Once the VM is
running, there are two ways it can access the data — "mount" vs.
"copy". Pick based on how your code reads files.

### Pattern A — Mount (recommended for this repo)

`gcsfuse` exposes the bucket as a normal directory. The kernel translates
every file read into a GCS API call under the hood. Nothing is
pre-downloaded; data streams on demand. Code that opens HDF5 files with
`h5py.File("/some/path/file.h5")` works with zero changes.

```bash
# One-time (on the VM):
./cloud/mount_bucket.sh ~/data/brain_val

# Then point scripts/reconstruct.py at the mount:
python scripts/reconstruct.py --mode edm \
  --checkpoint_dir checkpoints/edm/supervised_R=1 \
  --data_path ~/data/brain_val \
  --num_slices 5 --acceleration 4 --num_steps 20 \
  --schedule edm --sigma_max 10 --target_resolution 384 320

# When done:
fusermount -u ~/data/brain_val
```

Good for: sequential reads over a handful of slices (the typical
evaluation pattern here). No extra local disk needed.

Bad for: heavy random-access HDF5 workloads (thousands of seeks per
file) — gcsfuse adds per-read latency that adds up. If that becomes a
bottleneck, switch to Pattern B.

### Pattern B — Copy to local SSD first

Download the files you need onto the VM's disk, then read them as
regular local files. Faster random access, but you pay VM disk cost
for the duration.

```bash
# Extract and copy once, on a VM with enough disk (e.g. 150 GB SSD):
mkdir -p ~/data/brain_val
gcloud storage cp gs://fastmri-493921-brain-val/brain_multicoil_val_batch_0.tar.xz ~
tar -xJf ~/brain_multicoil_val_batch_0.tar.xz -C ~/data/brain_val
rm ~/brain_multicoil_val_batch_0.tar.xz

python scripts/reconstruct.py --mode edm --data_path ~/data/brain_val ...
```

Good for: repeated experiments over the same slices, random HDF5
access, notebook workflows where you re-open the same files many times.

Bad for: one-shot evals (wastes disk), laptops outside `us-central1`
(triggers egress fees).

### A note on laptops

Running directly from a laptop against the bucket works (just
`gcsfuse`-mount it or `gcloud storage cp` what you need) but each GB
read costs ~$0.12 in egress. Fine for a handful of slices during
development; expensive if you loop over all 94 GB. Spin up a cheap
`us-central1` VM instead whenever you're doing more than casual
inspection.

## Retention

The lifecycle rule deletes every object 90 days after a custom time of
**2026-06-12** — so everything is purged on **2026-09-10**, 90 days
after the course's poster session. `upload_from_nyu.sh` stamps every
upload with that custom time, so the deadline is exact regardless of
when the upload actually happens.

If the course timeline shifts, edit `LIFECYCLE_CUSTOM_TIME` in
`setup_bucket.sh` and re-run it; existing objects keep their stamp.

## Who has access

The setup script grants these roles, scoped as narrowly as possible:

| Identity | Role | Scope | Why |
|---|---|---|---|
| `carlops@stanford.edu` | `roles/storage.objectUser` | bucket | read/write data |
| `esampi@stanford.edu` | `roles/storage.objectUser` | bucket | read/write data |
| `carlops@stanford.edu` | `roles/compute.instanceAdmin.v1` | project | spin up training VMs |
| `esampi@stanford.edu` | `roles/compute.instanceAdmin.v1` | project | spin up training VMs |
| `carlops@stanford.edu` | `roles/iam.serviceAccountUser` | project | attach service accounts to VMs |
| `esampi@stanford.edu` | `roles/iam.serviceAccountUser` | project | attach service accounts to VMs |

To add a third collaborator, append their email to `COLLABORATORS` in
`setup_bucket.sh` and re-run it — the grants are idempotent.
