import math

import torch


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Normalized Mean Squared Error."""
    return torch.norm(target - pred) ** 2 / torch.norm(target) ** 2


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio."""
    mse = torch.mean((target - pred) ** 2)
    max_val = target.max()
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def stacked_psnr_from_sufficient_statistics(
    squared_error_sum: float,
    numel: int,
    target_max: float,
) -> float:
    """fastMRI-style PSNR for a stacked volume.

    The official evaluator computes one MSE and one target maximum over the
    complete ``(slice, height, width)`` volume. Keeping these sufficient
    statistics lets the reconstruction runner process slices independently
    without silently changing the metric grain.
    """
    if numel <= 0:
        raise ValueError("numel must be positive")
    if squared_error_sum < 0:
        raise ValueError("squared_error_sum must be nonnegative")
    if target_max <= 0:
        raise ValueError("target_max must be positive")
    if squared_error_sum == 0:
        return float("inf")
    mse = float(squared_error_sum) / int(numel)
    return float(10.0 * math.log10(float(target_max) ** 2 / mse))


def stacked_nmse_from_sufficient_statistics(
    squared_error_sum: float,
    target_squared_sum: float,
) -> float:
    """fastMRI-style NMSE for a stacked volume."""
    if squared_error_sum < 0:
        raise ValueError("squared_error_sum must be nonnegative")
    if target_squared_sum <= 0:
        raise ValueError("target_squared_sum must be positive")
    return float(squared_error_sum) / float(target_squared_sum)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    win_size: int = 7,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Structural Similarity Index (simplified single-channel version)."""
    C1 = (k1 * target.max()) ** 2
    C2 = (k2 * target.max()) ** 2

    # Use average pooling as a proxy for windowed mean
    kernel_size = win_size
    pad = win_size // 2

    # Add batch/channel dims if needed
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)

    mu_pred = torch.nn.functional.avg_pool2d(pred, kernel_size, stride=1, padding=pad)
    mu_target = torch.nn.functional.avg_pool2d(target, kernel_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = torch.nn.functional.avg_pool2d(pred ** 2, kernel_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = torch.nn.functional.avg_pool2d(target ** 2, kernel_size, stride=1, padding=pad) - mu_target_sq
    sigma_cross = torch.nn.functional.avg_pool2d(pred * target, kernel_size, stride=1, padding=pad) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return ssim_map.mean()
