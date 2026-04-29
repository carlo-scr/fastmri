#!/usr/bin/env bash
# Step 4: Volumetric ACS Pooling — main experiment sweep.
#
# Compares three σ²_i initialisations for FA-KGD on real data:
#   1. oracle      — true σ² (cheating upper bound)
#   2. pooled_acs  — proposed: ACS pooled across all slices of the volume
#   3. ΠGDM baseline (independent of noise_init) — already produced in every run
#
# Per-slice ACS is degenerate in the single-coil RSS pipeline (only one
# replicate per pixel), so we omit it; this is itself a paper-worthy point
# made in the new subsection.
#
# Each run processes EVERY slice of the first NUM_VOLS volumes so we can
# aggregate per-volume statistics.
#
# DO NOT RUN as-is in the foreground for the full sweep; this script just
# emits the commands. Adjust NUM_VOLS / NUM_STEPS for budget.
set -euo pipefail

cd "$(dirname "$0")/.."
source venv/bin/activate

DATA=data/multicoil_val
CKPT=checkpoints/edm/supervised_R=1
T=20                       # diffusion steps (T=20 is the project default)
NUM_VOLS=10                # volumes to process
# multicoil_val brain volumes have ~16 slices → ~160 slices per setting
NUM_SLICES=$((NUM_VOLS * 32))   # generous upper bound; --whole_volume controls actual count
DEVICE=cpu                 # bump to cuda if available

OUTROOT=outputs/acs_pooling_sweep_T${T}
mkdir -p "$OUTROOT"

run () {
  local R="$1" INIT="$2" MMODE="$3"
  local OUT="${OUTROOT}/R${R}_${INIT}"
  echo
  echo "=== R=${R}  noise_init=${INIT}  m_step=${MMODE}  → ${OUT} ==="
  python scripts/reconstruct.py \
    --mode edm \
    --checkpoint_dir "$CKPT" \
    --data_path "$DATA" \
    --num_slices "$NUM_SLICES" \
    --whole_volume \
    --acceleration "$R" \
    --num_steps "$T" \
    --schedule edm \
    --noise_init "$INIT" \
    --m_step_mode "$MMODE" \
    --target_resolution 384 320 \
    --device "$DEVICE" \
    --output_dir "$OUT"
}

for R in 4 8; do
  run "$R" oracle     clamp
  run "$R" pooled_acs clamp
done

echo
echo "Sweep complete. JSON summaries in $OUTROOT/*/results.json"
