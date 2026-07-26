"""Held-out ImageNet-256 active block acquisition with official ADM.

This is the scale-up natural-image experiment for SketchSense. It requires an
explicit ImageNet validation directory and fails closed when it is absent. No
training split, silent fallback, or cached unlabeled tensor is accepted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from poc_natural.guided_adm import (
    load_heldout_image_folder,
    load_imagenet256_unconditional_adm,
    validate_imagenet_validation_layout,
)
from poc_natural.pvas_core import (
    DenoiserStats,
    EpsDenoiser,
    PVASConfig,
    acquire_blocks,
    init_block_state,
    random_block_scores,
    run_pvas,
)
from poc_natural.run_cifar import (
    cuda_peak_memory_mb,
    lpips_batch,
    observed_block_ids,
    paired_bootstrap_mean_ci,
    psnr,
    reset_cuda_peak_memory,
    run_canonical_replay,
    run_ensemble_active,
    ssim_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True,
                        help="Official held-out ImageNet validation directory")
    parser.add_argument("--checkpoint", required=True,
                        help="OpenAI 256x256_diffusion_uncond.pt")
    parser.add_argument("--guided_repo", default="external/guided-diffusion")
    parser.add_argument("--n_images", type=int, default=250)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--budget_blocks", type=int, default=32)
    parser.add_argument("--active_rounds", type=int, default=2)
    parser.add_argument("--ensemble_k", type=int, default=4)
    parser.add_argument("--sketch_rank", type=int, default=4)
    parser.add_argument("--sketch_fd_step", type=float, default=5e-2)
    parser.add_argument("--sketch_psd_rtol", type=float, default=1e-6)
    parser.add_argument("--policies", nargs="+", default=[
        "static", "random", "operator_diag", "sketch_marginal",
        "sketch_logdet", "ensemble", "ens_random", "oracle",
    ])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--lpips",
        action="store_true",
        help="Also compute LPIPS-Alex for system and canonical reconstructions.",
    )
    parser.add_argument(
        "--skip_canonical_replay",
        action="store_true",
        help="Pilot-only speed option. Production selection comparisons should "
             "replay every final mask with the same passive solver.",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires CUDA")
    torch.manual_seed(args.seed)

    dataset_provenance = validate_imagenet_validation_layout(args.image_dir)
    images, sample_manifest = load_heldout_image_folder(
        args.image_dir,
        n_images=args.n_images,
        seed=args.seed + 17,
        image_size=256,
    )
    unet, abar, model_provenance = load_imagenet256_unconditional_adm(
        args.checkpoint,
        args.guided_repo,
        device=str(device),
        use_fp16=args.fp16,
    )
    lpips_metric = None
    if args.lpips:
        import lpips as lpips_library
        lpips_metric = lpips_library.LPIPS(net="alex").to(device).eval()

    grid_h = grid_w = 256 // args.block
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "status": "CORRECTED_RUN_UNREVIEWED",
        "config": vars(args),
        "protocol": {
            "dataset_role": "explicit held-out image folder",
            "primary_selection_metric": (
                "system reconstruction"
                if args.skip_canonical_replay
                else "canonical single-trajectory replay of the final mask"
            ),
            "system_and_selection_metrics_separated": True,
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else str(device)
            ),
        },
        "dataset_provenance": dataset_provenance,
        "model_provenance": model_provenance,
        "sample_manifest": sample_manifest,
        "git_commit": _git_commit(),
        "per_image": {},
        "summary": {},
        "nfe": {},
    }
    start = time.time()

    for policy in args.policies:
        system_psnr: list[float] = []
        system_ssim: list[float] = []
        selection_psnr: list[float] = []
        selection_ssim: list[float] = []
        system_lpips: list[float] = []
        selection_lpips: list[float] = []
        final_atom_ids: list[list[int]] = []
        batch_seeds: list[dict[str, int]] = []
        nfe_total = 0
        canonical_nfe_total = 0
        system_wall_time = 0.0
        canonical_wall_time = 0.0
        system_peak_memory_mb = None
        canonical_peak_memory_mb = None
        for batch_start in range(0, args.n_images, args.batch):
            x0 = images[batch_start: batch_start + args.batch].to(device)
            measurement_seed = args.seed + 1000 + batch_start
            sampler_seed = args.seed + batch_start
            measurement_generator = (
                torch.Generator(device="cpu").manual_seed(measurement_seed)
            )
            initial_state = init_block_state(
                x0,
                frac=0.0,
                block=args.block,
                sigma_y=0.0,
                gen=measurement_generator,
                with_lr=True,
            )
            config = PVASConfig(
                num_steps=args.num_steps,
                use_pv_gate=False,
                active_rounds=args.active_rounds,
                policy="random" if policy == "static" else policy,
                ensemble_k=args.ensemble_k,
                sketch_rank=args.sketch_rank,
                sketch_fd_step=args.sketch_fd_step,
                sketch_psd_rtol=args.sketch_psd_rtol,
                seed=sampler_seed,
            )

            reset_cuda_peak_memory(device)
            system_started_at = time.time()
            if policy == "static":
                scores = random_block_scores(
                    x0.shape[0],
                    grid_h,
                    grid_w,
                    config.seed,
                    device,
                )
                initial_state = acquire_blocks(
                    initial_state,
                    x0,
                    scores,
                    args.budget_blocks,
                    torch.Generator(device="cpu").manual_seed(
                        config.seed + 2_000_003
                    ),
                )
                stats = DenoiserStats()
                denoiser = EpsDenoiser(unet, abar, stats)
                static_config = PVASConfig(**{
                    **config.__dict__,
                    "active_rounds": 0,
                    "policy": "random",
                })
                result = run_pvas(
                    denoiser, initial_state, x0, 0, static_config
                )
                nfe_total += stats.nfe
            elif policy in {"ensemble", "ens_random"}:
                selection = "var" if policy == "ensemble" else "random"
                result, nfe = run_ensemble_active(
                    unet,
                    abar,
                    initial_state,
                    x0,
                    args.budget_blocks,
                    config,
                    args.ensemble_k,
                    select=selection,
                )
                nfe_total += nfe
            else:
                stats = DenoiserStats()
                denoiser = EpsDenoiser(unet, abar, stats)
                result = run_pvas(
                    denoiser,
                    initial_state,
                    x0,
                    args.budget_blocks,
                    config,
                )
                nfe_total += stats.nfe

            batch_system_peak = cuda_peak_memory_mb(device)
            system_wall_time += float(time.time() - system_started_at)
            if batch_system_peak is not None:
                system_peak_memory_mb = max(
                    system_peak_memory_mb or 0.0,
                    batch_system_peak,
                )
            system_psnr.extend(psnr(result.x0, x0).cpu().tolist())
            system_ssim.extend(ssim_batch(result.x0, x0).tolist())
            if lpips_metric is not None:
                system_lpips.extend(lpips_batch(lpips_metric, result.x0, x0))
            final_atom_ids.extend(observed_block_ids(result.state))
            batch_seeds.append({
                "batch_start": int(batch_start),
                "measurement_seed": int(measurement_seed),
                "sampler_seed": int(sampler_seed),
            })

            if args.skip_canonical_replay or policy == "static":
                selection_result = result
            else:
                reset_cuda_peak_memory(device)
                canonical_started_at = time.time()
                selection_result, replay_nfe = run_canonical_replay(
                    unet, abar, result.state, x0, config
                )
                canonical_nfe_total += replay_nfe
                batch_canonical_peak = cuda_peak_memory_mb(device)
                canonical_wall_time += float(
                    time.time() - canonical_started_at
                )
                if batch_canonical_peak is not None:
                    canonical_peak_memory_mb = max(
                        canonical_peak_memory_mb or 0.0,
                        batch_canonical_peak,
                    )
            selection_psnr.extend(
                psnr(selection_result.x0, x0).cpu().tolist()
            )
            selection_ssim.extend(
                ssim_batch(selection_result.x0, x0).tolist()
            )
            if lpips_metric is not None:
                selection_lpips.extend(
                    lpips_batch(lpips_metric, selection_result.x0, x0)
                )
            print(
                f"[{policy}] {batch_start + x0.shape[0]}/{args.n_images} "
                f"selection_PSNR={np.mean(selection_psnr):.3f} "
                f"system_PSNR={np.mean(system_psnr):.3f} "
                f"elapsed={time.time() - start:.0f}s",
                flush=True,
            )

        results["per_image"][policy] = {
            "psnr": selection_psnr,
            "ssim": selection_ssim,
            "system_psnr": system_psnr,
            "system_ssim": system_ssim,
            "final_atom_ids": final_atom_ids,
            "batch_seeds": batch_seeds,
        }
        if lpips_metric is not None:
            results["per_image"][policy].update({
                "lpips": selection_lpips,
                "system_lpips": system_lpips,
            })
        results["nfe"][policy] = {
            "system_nfe_per_image": nfe_total / args.n_images,
            "evaluation_only_canonical_replay_nfe_per_image": (
                canonical_nfe_total / args.n_images
            ),
        }
        results["summary"][policy] = {
            "psnr_mean": float(np.mean(selection_psnr)),
            "psnr_std": float(np.std(selection_psnr)),
            "ssim_mean": float(np.mean(selection_ssim)),
            "ssim_std": float(np.std(selection_ssim)),
            "system_psnr_mean": float(np.mean(system_psnr)),
            "system_psnr_std": float(np.std(system_psnr)),
            "system_ssim_mean": float(np.mean(system_ssim)),
            "system_ssim_std": float(np.std(system_ssim)),
            "system_nfe_per_image": nfe_total / args.n_images,
            "evaluation_only_canonical_replay_nfe_per_image": (
                canonical_nfe_total / args.n_images
            ),
            "system_wall_time_seconds": system_wall_time,
            "system_peak_cuda_memory_mb": system_peak_memory_mb,
            "evaluation_only_canonical_replay_wall_time_seconds": (
                canonical_wall_time
            ),
            "evaluation_only_canonical_replay_peak_cuda_memory_mb": (
                canonical_peak_memory_mb
            ),
            "n": len(selection_psnr),
        }
        if lpips_metric is not None:
            results["summary"][policy].update({
                "lpips_mean": float(np.mean(selection_lpips)),
                "lpips_std": float(np.std(selection_lpips)),
                "system_lpips_mean": float(np.mean(system_lpips)),
                "system_lpips_std": float(np.std(system_lpips)),
            })

    _paired_deltas(results)
    results["runtime_seconds"] = time.time() - start
    path = output_dir / "results.json"
    with path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"saved {path}", flush=True)


def _paired_deltas(results: dict):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return
    for candidate in ["operator_diag", "sketch_marginal", "sketch_logdet"]:
        if candidate not in results["per_image"]:
            continue
        values = np.asarray(results["per_image"][candidate]["psnr"])
        for reference in [
            "static", "random", "operator_diag", "sketch_marginal",
            "ensemble", "oracle",
        ]:
            if reference == candidate or reference not in results["per_image"]:
                continue
            difference = (
                values
                - np.asarray(results["per_image"][reference]["psnr"])
            )
            try:
                p_value = float(
                    wilcoxon(difference, alternative="greater").pvalue
                )
            except ValueError:
                p_value = float("nan")
            ci_low, ci_high = paired_bootstrap_mean_ci(
                difference,
                seed=91,
            )
            results["summary"][f"delta_{candidate}_vs_{reference}"] = {
                "mean": float(difference.mean()),
                "std": float(difference.std()),
                "paired_bootstrap_95ci": [ci_low, ci_high],
                "wilcoxon_p_greater": p_value,
            }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
