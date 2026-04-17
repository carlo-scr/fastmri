import argparse
import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SliceDataset
from src.data.transforms import (
    center_crop,
    create_mask,
    ifft2c,
    normalize,
    to_tensor,
)
from src.models.unet import UNet
from src.utils.metrics import nmse, psnr, ssim


def train_transform(kspace, target, fname, slice_idx):
    """Transform for training: undersample k-space -> zero-filled input, target."""
    kspace_t = to_tensor(kspace)  # (H, W, 2) complex as real/imag
    kspace_complex = torch.complex(kspace_t[..., 0], kspace_t[..., 1])

    # Create and apply undersampling mask
    mask = create_mask(kspace_complex.shape[-1], center_fraction=0.08, acceleration=4)
    masked_kspace = kspace_complex * mask

    # Zero-filled reconstruction via inverse FFT
    image = ifft2c(masked_kspace)
    image = torch.abs(image)

    # Crop and normalize
    image = center_crop(image, (320, 320))
    image = normalize(image)

    target_t = torch.from_numpy(target) if target is not None else image.clone()
    target_t = center_crop(target_t, (320, 320))
    target_t = normalize(target_t)

    return image.unsqueeze(0), target_t.unsqueeze(0)  # (1, H, W)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_data = SliceDataset(args.data_path / "singlecoil_train", transform=train_transform)
    val_data = SliceDataset(args.data_path / "singlecoil_val", transform=train_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = UNet(in_chans=1, out_chans=1, chans=args.chans, num_pool_layers=args.num_pool_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (input_img, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_img, target = input_img.to(device), target.to(device)

            output = model(input_img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():
            for input_img, target in val_loader:
                input_img, target = input_img.to(device), target.to(device)
                output = model(input_img)
                val_loss += criterion(output, target).item()
                val_psnr += psnr(output, target).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = args.output_path / "best_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_loss": val_loss}, save_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for MRI reconstruction")
    parser.add_argument("--data-path", type=pathlib.Path, default="data", help="Path to data directory")
    parser.add_argument("--output-path", type=pathlib.Path, default="outputs", help="Path for checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chans", type=int, default=32, help="Base U-Net channels")
    parser.add_argument("--num-pool-layers", type=int, default=4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
