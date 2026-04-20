#!/usr/bin/env bash
# Create and configure the private GCS bucket that hosts the fastMRI
# brain validation data for this project. Idempotent: safe to re-run.
#
# Required roles for the caller: project Owner or Storage Admin + IAM
# Admin. Run from any machine with gcloud installed and authenticated.

set -euo pipefail

PROJECT="${PROJECT:-fastmri-493921}"
BUCKET="${BUCKET:-fastmri-493921-brain-val}"
REGION="${REGION:-us-central1}"
LIFECYCLE_CUSTOM_TIME="${LIFECYCLE_CUSTOM_TIME:-2026-06-12}"
LIFECYCLE_DAYS_AFTER="${LIFECYCLE_DAYS_AFTER:-90}"

COLLABORATORS=(
  "carlops@stanford.edu"
  "esampi@stanford.edu"
)

echo "Project:      $PROJECT"
echo "Bucket:       gs://$BUCKET"
echo "Region:       $REGION"
echo "Delete-after: $LIFECYCLE_DAYS_AFTER days past customTime $LIFECYCLE_CUSTOM_TIME"
echo

gcloud config set project "$PROJECT" >/dev/null

# Bucket: create if missing, enforce private access settings.
if gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo "Bucket already exists; reapplying policy."
else
  gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

gcloud storage buckets update "gs://$BUCKET" \
  --uniform-bucket-level-access \
  --public-access-prevention

# Lifecycle: delete every object N days after its customTime stamp.
# upload_from_nyu.sh stamps each upload with LIFECYCLE_CUSTOM_TIME, so
# the effective deletion date is fixed regardless of upload timing.
LIFECYCLE_JSON=$(mktemp)
trap 'rm -f "$LIFECYCLE_JSON"' EXIT
cat >"$LIFECYCLE_JSON" <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"daysSinceCustomTime": $LIFECYCLE_DAYS_AFTER}
      }
    ]
  }
}
EOF
gcloud storage buckets update "gs://$BUCKET" --lifecycle-file="$LIFECYCLE_JSON"

# IAM: bucket-scoped read/write for each collaborator, plus project-level
# compute permissions so they can run training VMs.
for email in "${COLLABORATORS[@]}"; do
  echo "Granting access to $email"
  gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
    --member="user:$email" \
    --role="roles/storage.objectUser" >/dev/null
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="user:$email" \
    --role="roles/compute.instanceAdmin.v1" \
    --condition=None >/dev/null
  gcloud projects add-iam-policy-binding "$PROJECT" \
    --member="user:$email" \
    --role="roles/iam.serviceAccountUser" \
    --condition=None >/dev/null
done

echo
echo "Done. gs://$BUCKET is ready."
