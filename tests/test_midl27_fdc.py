"""Correctness checks for experiments/midl27/fdc.py (CPU, no model needed)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "midl27"))
import fdc  # noqa: E402


def _crandn(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g)).to(torch.complex128)


def test_single_coil_cg_matches_closed_form_kalman():
    H, W = 32, 24
    mask = fdc.random_mask(W, 4, 0.2, seed=1)[None].expand(H, W).double()
    sens = torch.ones(1, H, W, dtype=torch.complex128)
    op = fdc.SenseOp(mask, sens)
    rho = fdc.radius_grid(H, W).double()
    p = 1.0 / (1.0 + rho) ** 2          # strongly non-white
    s2 = 0.05
    r = mask * _crandn(1, H, W, seed=2)
    K = p / (p + s2)
    expected = fdc.ifft2c(mask * K * r[0])
    delta = fdc.dc_solve(op, r, p, s2, n_iter=5, precond=True)
    assert torch.allclose(delta, expected, atol=1e-8)
    plain = fdc.dc_solve(op, r, p, s2, n_iter=400, tol=1e-14)
    assert torch.allclose(plain, expected, atol=1e-7)


def test_multicoil_pcg_matches_dense_solve():
    H, W, Nc = 12, 10, 3
    mask = fdc.random_mask(W, 2, 0.3, seed=3)[None].expand(H, W).double()
    sens = _crandn(Nc, H, W, seed=4) * 0.5
    op = fdc.SenseOp(mask, sens)
    rho = fdc.radius_grid(H, W).double()
    p = 2.0 / (1.0 + 0.3 * rho)
    s2 = 0.1
    r = mask * _crandn(Nc, H, W, seed=5)
    delta = fdc.dc_solve(op, r, p, s2, n_iter=400, tol=1e-14)
    n = H * W
    E = torch.eye(n, dtype=torch.complex128).reshape(n, H, W)
    cols = [(op.AH(op.A(e)) + fdc.ifft2c((s2 / p) * fdc.fft2c(e))).flatten() for e in E]
    M = torch.stack(cols, 1)
    expected = torch.linalg.solve(M, op.AH(r).flatten()).reshape(H, W)
    assert torch.allclose(delta, expected, atol=1e-7)


def test_spectral_variance_reduces_to_iso_for_large_spectrum():
    rho = fdc.radius_grid(8, 8)
    spec = fdc.Spectrum(edges=np.linspace(0, 10, 3), binned=np.ones(2), coef=(np.log(1e12), 0.0),
                        dc=1e12, floor=1e-12)
    p_spec = fdc.posterior_variance("spectral", 0.7, rho, spectrum=spec)
    p_iso = fdc.posterior_variance("iso", 0.7, rho)
    assert torch.allclose(p_spec, p_iso, rtol=1e-6)


def test_white_equals_spectral_with_flat_spectrum():
    rho = fdc.radius_grid(8, 8)
    spec = fdc.Spectrum(edges=np.linspace(0, 10, 3), binned=np.ones(2), coef=(np.log(3.0), 0.0),
                        dc=3.0, floor=1e-12)
    a = fdc.posterior_variance("spectral", 0.5, rho, spectrum=spec)
    b = fdc.posterior_variance("white", 0.5, rho, data_var=3.0)
    assert torch.allclose(a, b, rtol=1e-6)


def test_equispaced_mask_budget():
    for R, cf in ((4, 0.08), (8, 0.04)):
        m = fdc.equispaced_mask(320, R, cf, seed=0)
        assert abs(int(m.sum()) - 320 / R) <= 3


def test_whitening_identity():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    psi = A @ A.conj().T + 0.1 * np.eye(4)
    Wm = fdc.whitening_matrix(psi)
    assert np.allclose(Wm @ psi @ Wm.conj().T, np.eye(4), atol=1e-8)


def test_spectrum_recovers_power_law_slope():
    H = W = 128
    rho = fdc.radius_grid(H, W).double().clamp(min=1.0)
    true_b = 2.5
    S = 1e4 * rho ** (-true_b)
    x_k = torch.sqrt(S) * _crandn(H, W, seed=6) / np.sqrt(2)
    s2 = 1e-3
    y = (x_k + np.sqrt(s2) * _crandn(H, W, seed=7) / np.sqrt(2))[None]
    mask = torch.ones(H, W, dtype=torch.float64)
    spec = fdc.estimate_spectrum(y, mask, s2)
    assert abs(spec.coef[1] - true_b) < 0.3
