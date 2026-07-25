#!/bin/zsh
# Second MRI regime: sigma_max=80 (the validate_active.py configuration).
# 10 brain volumes x 3 slices; writes outputs/PVAS_local80/<name>/results.json
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON=(--data_path data/brain_val --checkpoint_dir "checkpoints/edm/supervised_R=1"
        --num_volumes 10 --num_slices 3 --num_steps 20 --sigma_max 80
        --target_resolution 384 320 --device mps --active_rounds 2)

echo "=== config E: R4 L8 smax80 ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 4 --active_lines 8 \
  --methods pigdm_mc pv pv_active random_adaptive static_match oracle_active \
  --output_dir outputs/PVAS_local80/R4_a8

echo "=== config F: R8 L16 smax80 ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 8 --active_lines 16 \
  --methods pigdm_mc pv pv_active random_adaptive static_match oracle_active \
  --output_dir outputs/PVAS_local80/R8_a16

echo "SMAX80 GRID DONE"
