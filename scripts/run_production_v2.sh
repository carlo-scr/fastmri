#!/bin/zsh
# Production v2 chain (gate-off selection-only PVAS + ensemble head-to-heads).
# Priority order: headline tables -> duo -> CelebA quant -> sigma80 -> 192-slice.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
M="pigdm_mc pv_active random_adaptive static_match oracle_active ens_active ens_random_active"
COMMON=(--checkpoint_dir "checkpoints/edm/supervised_R=1" --num_steps 20
        --target_resolution 384 320 --device mps --active_rounds 2
        --active_gate off --ens_particles 4)

echo "=== [A] 30-slice gate-off grid, sigma_max=10 ==="
for cfg in "4 8" "4 16" "8 8" "8 16"; do
  set -- $=cfg
  $PY scripts/reconstruct_mc.py "${COMMON[@]}" --data_path data/brain_val \
    --num_volumes 10 --num_slices 3 --sigma_max 10 \
    --acceleration $1 --active_lines $2 --methods $=M \
    --output_dir outputs/PVAS_ng/R${1}_a${2}
done

echo "=== [B] duo + production inpaint/SR reruns on CIFAR ==="
$PY -m poc_natural.run_cifar --mode inpaint --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.10 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \
  --policies static random pv duo oracle ensemble ens_random --tag inpaint_v2
$PY -m poc_natural.run_cifar --mode sr --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \
  --policies duo --tag sr_duo

echo "=== [C] CelebA-HQ 256 quantitative (n=64) ==="
$PY -m poc_natural.quantitative_celeba --n_images 64 --batch 4 --num_steps 100 \
  --budget_blocks 96 --active_rounds 3 --ensemble_k 4 \
  --policies static random pv oracle duo ensemble --tag quant

echo "=== [D] sigma_max=80 gate-off (2 headline configs) ==="
for cfg in "4 8" "8 16"; do
  set -- $=cfg
  $PY scripts/reconstruct_mc.py "${COMMON[@]}" --data_path data/brain_val \
    --num_volumes 10 --num_slices 3 --sigma_max 80 \
    --acceleration $1 --active_lines $2 --methods $=M \
    --output_dir outputs/PVAS_ng80/R${1}_a${2}
done

echo "=== [E] 192-slice production (12 AXFLAIR vols, chunked) ==="
for chunk in 1 2 3; do
  for cfg in "8 16" "4 8"; do
    set -- $=cfg
    $PY scripts/reconstruct_mc.py "${COMMON[@]}" --data_path data/axflair_chunk$chunk \
      --num_volumes 4 --num_slices 0 --sigma_max 10 \
      --acceleration $1 --active_lines $2 --methods $=M \
      --output_dir outputs/PVAS_full/R${1}_a${2}_chunk${chunk}
  done
done

echo "PRODUCTION V2 CHAIN DONE"
