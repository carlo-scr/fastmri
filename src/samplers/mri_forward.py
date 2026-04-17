"""MRI forward models for single-coil and multi-coil acquisitions."""

import torch
import numpy as np


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT (ortho-normalized). Works on complex tensors."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def ifft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse FFT (ortho-normalized). Works on complex tensors."""
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    x = torch.fft.fftshift(x, dim=(-2, -1))
    return x


def build_radius_grid(H: int, W: int) -> torch.Tensor:
    """Build a grid of normalized frequency radii for FPDC.

    Returns a (H, W) tensor where each entry is the distance from DC center,
    normalized so the maximum is 1.0.
    """
    fy = torch.arange(H, dtype=torch.float32) - H // 2
    fx = torch.arange(W, dtype=torch.float32) - W // 2
    gy, gx = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(gy**2 + gx**2)
    return radius


def create_mask(
    num_cols: int,
    center_fraction: float = 0.08,
    acceleration: int = 4,
    seed: int = 42,
) -> torch.Tensor:
    """Create a 1D random undersampling mask (1, W) for Cartesian k-space."""
    rng = np.random.RandomState(seed)
    num_low_freqs = int(round(num_cols * center_fraction))
    mask = np.zeros(num_cols, dtype=np.float32)
    center = num_cols // 2
    mask[center - num_low_freqs // 2 : center + num_low_freqs // 2] = 1
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs + 1e-8)
    other_mask = rng.uniform(size=num_cols) < prob
    mask = np.maximum(mask, other_mask.astype(np.float32))
    return torch.from_numpy(mask).unsqueeze(0)  # (1, W)


class SingleCoilMRI:
    """Single-coil MRI forward model: A = M · F (mask · FFT)."""

    def __init__(self, mask: torch.Tensor):
        """
        Args:
            mask: k-space undersampling mask, broadcastable to (H, W).
        """
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (complex, [..., H, W]) → undersampled k-space."""
        return self.mask * fft2c(x)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Undersampled k-space → image estimate (zero-filled recon)."""
        return ifft2c(self.mask * y)

    def to(self, device):
        self.mask = self.mask.to(device)
        return self


class MultiCoilMRI:
    """Multi-coil MRI forward model: A = M · F · S (mask · FFT · coil maps)."""

    def __init__(self, mask: torch.Tensor, sensitivity_maps: torch.Tensor):
        """
        Args:
            mask: k-space mask, broadcastable to (H, W).
            sensitivity_maps: complex coil maps (num_coils, H, W).
        """
        self.mask = mask
        self.maps = sensitivity_maps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (complex, [B, H, W] or [H, W]) → multi-coil undersampled k-space."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        coil_imgs = self.maps * x.unsqueeze(-3)  # (B, C, H, W)
        return self.mask * fft2c(coil_imgs)

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Multi-coil k-space → combined image."""
        coil_imgs = ifft2c(self.mask * y)
        return (torch.conj(self.maps) * coil_imgs).sum(dim=-3)

    def to(self, device):
        self.mask = self.mask.to(device)
        self.maps = self.maps.to(device)
        return self
