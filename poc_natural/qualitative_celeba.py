"""CelebA-HQ 256 qualitative panel: guided super-resolution with active blocks.

LR 32x32 preview always observed (8x box average); the budget buys 8x8-pixel
full-resolution blocks. Policies: static (random up-front), random (mid), pv
(ours), oracle. Saves per-policy reconstructions, the PV map, and acquired-block
overlays; renders paper/pvas_v2/figs/fig_qual_sr.pdf.

Run: python -m poc_natural.qualitative_celeba --n_images 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poc_natural.pvas_core import (  # noqa: E402
    DenoiserStats, EpsDenoiser, PVASConfig, acquire_blocks, init_block_state,
    make_abar, run_pvas,
)
from poc_natural.run_cifar import psnr  # noqa: E402

CANDIDATE_DATASETS = [
    ("korexyz/celeba-hq-256x256", "image"),
    ("nielsr/CelebA-faces", "image"),
]


def load_faces(n: int) -> torch.Tensor:
    from datasets import load_dataset
    last_err = None
    for name, col in CANDIDATE_DATASETS:
        try:
            ds = load_dataset(name, split="train", streaming=True)
            imgs = []
            for ex in ds:
                im = ex[col].convert("RGB").resize((256, 256))
                imgs.append(np.array(im))
                if len(imgs) >= n:
                    break
            if len(imgs) >= n:
                print(f"loaded {n} faces from {name}")
                t = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
                return t / 255.0 * 2 - 1
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(f"no CelebA-HQ dataset loadable: {last_err}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_images", type=int, default=6)
    p.add_argument("--batch", type=int, default=3)
    p.add_argument("--num_steps", type=int, default=100)
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--budget_blocks", type=int, default=96)
    p.add_argument("--active_rounds", type=int, default=3)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device).eval()
    abar = make_abar(DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")).to(device)

    images = load_faces(args.n_images)
    out_dir = (Path(__file__).resolve().parents[1] / "outputs" / "poc_natural"
               / "celeba_qual")
    out_dir.mkdir(parents=True, exist_ok=True)

    gh = 256 // args.block
    recons, states, pvmaps = {}, {}, {}
    metrics = {}
    for pol in ["static", "random", "pv", "oracle"]:
        outs, msks, ps = [], [], []
        for b0 in range(0, args.n_images, args.batch):
            x0 = images[b0: b0 + args.batch].to(device)
            gen = torch.Generator().manual_seed(args.seed + 100 + b0)
            st = init_block_state(x0, 0.0, args.block, 0.0, gen, with_lr=True)
            cfg = PVASConfig(num_steps=args.num_steps, use_pv_gate=False,
                             active_rounds=args.active_rounds,
                             active_window=(0.25, 0.65), pv_n_probes=4,
                             policy=pol if pol != "static" else "random",
                             seed=args.seed + b0)
            if pol == "static":
                g2 = torch.Generator().manual_seed(args.seed + 999 + b0)
                sc = torch.rand(x0.shape[0], 1, gh, gh, generator=g2).to(device)
                st = acquire_blocks(st, x0, sc, args.budget_blocks, g2)
                cfg = PVASConfig(**{**cfg.__dict__, "active_rounds": 0})
                res = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0, 0, cfg)
            else:
                res = run_pvas(EpsDenoiser(unet, abar, DenoiserStats()), st, x0,
                               args.budget_blocks, cfg)
            outs.append(res.x0.cpu())
            msks.append(res.state.mask.cpu())
            if pol == "pv" and res.pv_maps:
                ps.append(res.pv_maps[-1][1])
        recons[pol] = torch.cat(outs)
        states[pol] = torch.cat(msks)
        if ps:
            pvmaps[pol] = torch.cat(ps)
        pv = psnr(recons[pol], images).tolist()
        metrics[pol] = {"psnr": pv, "psnr_mean": float(np.mean(pv))}
        print(f"{pol}: PSNR {np.mean(pv):.2f}", flush=True)

    torch.save({"images": images, "recons": recons, "masks": states,
                "pvmaps": pvmaps}, out_dir / "panel_data.pt")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"config": vars(args), "metrics": metrics}, f, indent=2)

    # ---- render panel ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = args.n_images
    cols = ["GT", "LR preview", "static", "random", "pv (ours)", "PV map + acq"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.1 * len(cols), 2.1 * n))
    to01 = lambda x: ((x.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    lr = torch.nn.functional.avg_pool2d(images, args.block)
    lr_up = torch.nn.functional.interpolate(lr, scale_factor=args.block, mode="nearest")
    for i in range(n):
        panels = [images[i], lr_up[i], recons["static"][i], recons["random"][i],
                  recons["pv"][i]]
        for j, im in enumerate(panels):
            axes[i, j].imshow(to01(im.cpu()))
        if "pv" in pvmaps and i < pvmaps["pv"].shape[0]:
            axes[i, 5].imshow(pvmaps["pv"][i, 0].numpy(), cmap="magma")
            m = states["pv"][i, 0].numpy()
            axes[i, 5].contour(m, levels=[0.5], colors="cyan", linewidths=0.6)
        for j in range(len(cols)):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(cols[j], fontsize=9)
    fig.tight_layout()
    fig_path = (Path(__file__).resolve().parents[1] / "paper" / "pvas_v2" / "figs"
                / "fig_qual_sr.pdf")
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=250, bbox_inches="tight")
    fig.savefig(fig_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print("saved", fig_path)


if __name__ == "__main__":
    main()
