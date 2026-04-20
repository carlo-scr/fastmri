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

## Using the data on a VM

On a Compute Engine VM with a GPU, mount the bucket with `gcsfuse` and
point `scripts/reconstruct.py` at the mount:

```bash
./cloud/mount_bucket.sh ~/data/brain_val
python scripts/reconstruct.py --mode edm \
  --checkpoint_dir checkpoints/edm/supervised_R=1 \
  --data_path ~/data/brain_val \
  --num_slices 5 --acceleration 4 --num_steps 20 \
  --schedule edm --sigma_max 10 --target_resolution 384 320
```

Unmount with `fusermount -u ~/data/brain_val` when done.

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
