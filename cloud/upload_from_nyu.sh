#!/usr/bin/env bash
# Stream a presigned fastMRI URL from NYU's S3 straight into the project
# bucket. Nothing is written to local disk; works from Cloud Shell, a
# Compute Engine VM, or a laptop.
#
# Usage:
#   export FASTMRI_URL='https://fastmri-dataset.s3.amazonaws.com/v2.0/...'
#   ./cloud/upload_from_nyu.sh "$FASTMRI_URL" brain_multicoil_val_batch_0.tar.xz
#
# Presigned URLs expire. Get fresh ones from https://fastmri.med.nyu.edu.

set -euo pipefail

BUCKET="${BUCKET:-fastmri-493921-brain-val}"
LIFECYCLE_CUSTOM_TIME="${LIFECYCLE_CUSTOM_TIME:-2026-06-12}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <presigned-url> <object-name>" >&2
  exit 2
fi

URL="$1"
OBJECT="$2"

echo "Streaming to gs://$BUCKET/$OBJECT"
echo "Custom time: ${LIFECYCLE_CUSTOM_TIME}T00:00:00Z (lifecycle anchor)"

curl --fail --location --silent --show-error "$URL" \
  | gcloud storage cp - "gs://$BUCKET/$OBJECT" \
      --custom-time="${LIFECYCLE_CUSTOM_TIME}T00:00:00Z"

echo "Uploaded gs://$BUCKET/$OBJECT"
gcloud storage ls -l "gs://$BUCKET/$OBJECT"
