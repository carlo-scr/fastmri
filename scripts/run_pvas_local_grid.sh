#!/bin/zsh
# Local (MPS) PVAS production grid: 10 brain volumes x 3 slices = 30 slices.
# Sequential configs; each writes outputs/PVAS_local/<name>/results.json
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
COMMON=(--data_path data/brain_val --checkpoint_dir "checkpoints/edm/supervised_R=1"
        --num_volumes 10 --num_slices 3 --num_steps 20 --sigma_max 10
        --target_resolution 384 320 --device mps --active_rounds 2)

echo "=== config A: R4 L8 (full method set) ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 4 --active_lines 8 \
  --methods pigdm_mc pv pv_active random_adaptive static_match oracle_active \
  --output_dir outputs/PVAS_local/R4_a8

echo "=== config B: R4 L16 ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 4 --active_lines 16 \
  --methods pv_active random_adaptive static_match \
  --output_dir outputs/PVAS_local/R4_a16

echo "=== config C: R8 L16 (full method set) ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 8 --active_lines 16 \
  --methods pigdm_mc pv pv_active random_adaptive static_match oracle_active \
  --output_dir outputs/PVAS_local/R8_a16

echo "=== config D: R8 L8 ==="
$PY scripts/reconstruct_mc.py "${COMMON[@]}" --acceleration 8 --active_lines 8 \
  --methods pv_active random_adaptive static_match \
  --output_dir outputs/PVAS_local/R8_a8

echo "ALL CONFIGS DONE"
