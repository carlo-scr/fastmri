#!/usr/bin/env python3
"""Quick hyperparameter sweep for FA-KGD on brain data (2 slices for speed)."""
import subprocess, sys, json, itertools
from pathlib import Path

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent.parent

# Sweep grid
betas  = [0.5, 1.0, 2.0]
alphas = [0.8, 0.9, 0.95, 1.0]

# Fixed params
COMMON = [
    PYTHON, "scripts/reconstruct.py",
    "--mode", "edm", "--schedule", "edm",
    "--num_steps", "20",
    "--checkpoint_dir", str(BASE_DIR / "checkpoints/edm/supervised_R=1"),
    "--data_path", str(BASE_DIR / "data/brain_val"),
    "--target_resolution", "384", "320",
    "--num_slices", "2",
    "--acceleration", "4",
    "--sigma_max", "10.0",
    "--device", "cpu",
]

results = []
for beta, alpha in itertools.product(betas, alphas):
    tag = f"sweep_b{beta}_a{alpha}"
    out_dir = str(BASE_DIR / f"outputs/{tag}")
    cmd = COMMON + [
        "--beta_fpdc", str(beta),
        "--alpha_ema", str(alpha),
        "--output_dir", out_dir,
    ]
    print(f"\n{'='*60}")
    print(f"  β={beta}, α={alpha}")
    print(f"{'='*60}")
    subprocess.run(cmd, cwd=str(BASE_DIR))

    # Read results
    rpath = Path(out_dir) / "results.json"
    if rpath.exists():
        with open(rpath) as f:
            s = json.load(f).get("summary", {})
        results.append({
            "beta": beta, "alpha": alpha,
            "pigdm": s.get("pigdm_mean_psnr", 0),
            "fakgd": s.get("fakgd_mean_psnr", 0),
            "delta": s.get("delta_psnr_mean", 0),
        })

# Summary
print(f"\n{'='*60}")
print(f"  SWEEP SUMMARY (2 slices, R=4, T=20)")
print(f"{'='*60}")
print(f"{'β':>6s} {'α':>6s} {'ΠGDM':>8s} {'FA-KGD':>8s} {'Δ PSNR':>8s}")
print("-" * 40)
best = None
for r in sorted(results, key=lambda x: -x["delta"]):
    marker = ""
    if best is None:
        best = r
        marker = " ← best"
    print(f"{r['beta']:>6.1f} {r['alpha']:>6.2f} {r['pigdm']:>7.2f}  {r['fakgd']:>7.2f}  {r['delta']:>+7.2f}{marker}")

if best:
    print(f"\nBest: β={best['beta']}, α={best['alpha']} → Δ={best['delta']:+.2f} dB")
    # Save summary
    with open(BASE_DIR / "outputs/sweep_summary.json", "w") as f:
        json.dump({"grid": results, "best": best}, f, indent=2)
    print(f"Saved to outputs/sweep_summary.json")
