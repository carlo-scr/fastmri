"""Analytic validation of the Tweedie second-moment PV estimator.

For a Gaussian prior x0 ~ N(m, diag(s2)) the MMSE denoiser is linear and the
posterior variance has closed form  (s2 * sig2) / (s2 + sig2)  with
sig2 = (1-abar)/abar.  The Hutchinson estimator in poc_natural.pvas_core must
recover it.  Run:  python -m poc_natural.sanity_gaussian
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poc_natural.pvas_core import pv_estimate, DenoiserStats  # noqa: E402


class GaussianDenoiser:
    """Exact MMSE denoiser for x0 ~ N(m, diag(s2)) under VP noising."""

    def __init__(self, m: torch.Tensor, s2: torch.Tensor, abar: torch.Tensor):
        self.m, self.s2, self.abar = m, s2, abar
        self.stats = DenoiserStats()

    def x0hat(self, xt: torch.Tensor, t: int) -> torch.Tensor:
        a = self.abar[t]
        sig2 = (1.0 - a) / a
        xtilde = xt / torch.sqrt(a)
        self.stats.nfe += xt.shape[0]
        return self.m + (self.s2 / (self.s2 + sig2)) * (xtilde - self.m)


def main():
    torch.manual_seed(0)
    B, C, H, W = 4, 3, 32, 32
    # smooth spatially varying prior variance in [0.05, 1.0]
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    s2 = (0.05 + 0.95 * (0.5 + 0.5 * torch.sin(6 * xx) * torch.cos(4 * yy)) ** 2)
    s2 = s2.view(1, 1, H, W).expand(1, C, H, W).contiguous()
    m = torch.zeros(1, C, H, W)

    abar = torch.linspace(0.9999, 1e-4, 1000)
    den = GaussianDenoiser(m, s2, abar)

    print(f"{'t':>5} {'sig2':>9} {'corr':>7} {'ratio':>7} {'relMAE':>8}")
    for t in [100, 300, 500, 700, 900]:
        a = abar[t]
        sig2 = ((1 - a) / a).item()
        analytic = (s2 * sig2 / (s2 + sig2)).mean(dim=1, keepdim=True)  # channel-avg
        x0 = m + torch.sqrt(s2) * torch.randn(B, C, H, W)
        xt = torch.sqrt(a) * x0 + torch.sqrt(1 - a) * torch.randn(B, C, H, W)
        est = pv_estimate(den, xt, t, eps_probe=5e-2, n_probes=8, smooth_sigma=None)
        est_m = est.mean(dim=0, keepdim=True)
        an = analytic.expand_as(est_m)
        corr = torch.corrcoef(torch.stack([est_m.flatten(), an.flatten()]))[0, 1].item()
        ratio = (est_m.mean() / an.mean()).item()
        rel_mae = ((est_m - an).abs().mean() / an.mean()).item()
        print(f"{t:>5} {sig2:>9.4f} {corr:>7.4f} {ratio:>7.4f} {rel_mae:>8.4f}")
    print("\nPASS if corr ~> 0.95 and ratio ~ 1.0 at mid/low t (estimator is exact "
          "for linear denoisers up to Hutchinson noise; more probes -> tighter).")


if __name__ == "__main__":
    main()
