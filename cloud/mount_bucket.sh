#!/usr/bin/env bash
# Mount the project bucket on a Compute Engine VM using gcsfuse, so
# scripts/reconstruct.py can read HDF5 volumes with no code changes.
#
# Usage:
#   ./cloud/mount_bucket.sh ~/data/brain_val
#   fusermount -u ~/data/brain_val   # when done

set -euo pipefail

BUCKET="${BUCKET:-fastmri-493921-brain-val}"
MOUNT_POINT="${1:-$HOME/data/brain_val}"

if ! command -v gcsfuse >/dev/null; then
  echo "gcsfuse not found. Install with:" >&2
  echo "  export GCSFUSE_REPO=gcsfuse-\$(lsb_release -c -s)" >&2
  echo "  echo \"deb https://packages.cloud.google.com/apt \$GCSFUSE_REPO main\" | sudo tee /etc/apt/sources.list.d/gcsfuse.list" >&2
  echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -" >&2
  echo "  sudo apt-get update && sudo apt-get install -y gcsfuse" >&2
  exit 1
fi

mkdir -p "$MOUNT_POINT"

# --implicit-dirs: treat GCS prefixes as directories (needed because we
# upload objects by name without explicit dir markers).
# --file-mode 0444: HDF5 files are read-only from the sampler's view.
gcsfuse --implicit-dirs --file-mode=0444 "$BUCKET" "$MOUNT_POINT"

echo "Mounted gs://$BUCKET at $MOUNT_POINT"
ls -la "$MOUNT_POINT" | head
