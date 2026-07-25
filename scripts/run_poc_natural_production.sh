#!/bin/zsh
# Production natural-image experiment suite (runs sequentially on MPS).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "=== [1/6] SR main table: n=128, budget 12, 7 policies ==="
$PY -m poc_natural.run_cifar --mode sr --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \
  --policies static random raster pv oracle ensemble ens_random --tag sr_main

echo "=== [2/6] SR budget sweep: n=64, budgets 6/18/24 ==="
for BUD in 6 18 24; do
  $PY -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \
    --frac0 0.0 --budget_blocks $BUD --sigma_y 0.0 --active_rounds 3 \
    --policies static random pv oracle --tag sr_b$BUD
done

echo "=== [3/6] SR noisy + gate ablation: sigma_y=0.1, gate on vs off ==="
$PY -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.1 --active_rounds 3 \
  --policies static random pv oracle --tag sr_noisy_nogate
$PY -m poc_natural.run_cifar --mode sr --n_images 64 --batch 16 --num_steps 100 \
  --frac0 0.0 --budget_blocks 12 --sigma_y 0.1 --active_rounds 3 --pv_gate \
  --policies pv --tag sr_noisy_gate

echo "=== [4/6] Inpaint main table (honest negative): n=128 ==="
$PY -m poc_natural.run_cifar --mode inpaint --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.10 --budget_blocks 12 --sigma_y 0.0 --active_rounds 3 --ensemble_k 8 \
  --policies static random pv oracle ensemble ens_random --tag inpaint_main

echo "=== [5/6] Calibration, SR mode: n=128, probes 1/2/4/8/16, ens 4/8/16 ==="
$PY -m poc_natural.calibration --mode sr --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.0 --sigma_y 0.0 --stop_frac 0.5 --probes 1 2 4 8 16 --ensemble_ks 4 8 16

echo "=== [6/6] Calibration, inpaint mode: n=128 ==="
$PY -m poc_natural.calibration --mode inpaint --n_images 128 --batch 16 --num_steps 100 \
  --frac0 0.10 --sigma_y 0.0 --stop_frac 0.5 --probes 1 2 4 8 16 --ensemble_ks 4 8 16

echo "ALL NATURAL-IMAGE PRODUCTION RUNS DONE"
