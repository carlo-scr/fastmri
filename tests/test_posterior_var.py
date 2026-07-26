import torch

from src.samplers.posterior_var import estimate_posterior_variance_kspace


def test_complex_identity_denoiser_has_calibrated_fourier_variance():
    """Each complex coordinate contains two real coordinates of variance σ²."""
    sigma = 0.7
    x = torch.randn(8, 6) + 1j * torch.randn(8, 6)

    def identity_denoiser(value, _sigma):
        return value

    variance, _ = estimate_posterior_variance_kspace(
        x_t=x,
        mu=x,
        sigma_t=sigma,
        denoiser_fn=identity_denoiser,
        eps_probe=1e-3,
        centered=True,
        radial_smooth=False,
        n_probes=1,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )

    expected = torch.full_like(variance, 2.0 * sigma**2)
    torch.testing.assert_close(variance, expected, rtol=2e-4, atol=2e-4)


def test_cross_hutchinson_does_not_square_linear_spectrum():
    """Regression for the former |Jv|² row-energy implementation."""
    sigma = 0.5
    scale = 3.0
    x = torch.randn(5, 5) + 1j * torch.randn(5, 5)

    def linear_denoiser(value, _sigma):
        return scale * value

    variance, _ = estimate_posterior_variance_kspace(
        x_t=x,
        mu=linear_denoiser(x, sigma),
        sigma_t=sigma,
        denoiser_fn=linear_denoiser,
        eps_probe=1e-3,
        centered=True,
        radial_smooth=False,
        n_probes=1,
        generator=torch.Generator(device="cpu").manual_seed(11),
    )

    # True total complex variance is two real channels times σ² J.
    expected = torch.full_like(variance, 2.0 * sigma**2 * scale)
    wrong_squared_row_energy = 2.0 * sigma**2 * scale**2
    torch.testing.assert_close(variance, expected, rtol=3e-4, atol=3e-4)
    assert not torch.allclose(
        variance,
        torch.full_like(variance, wrong_squared_row_energy),
        rtol=1e-2,
        atol=1e-2,
    )
