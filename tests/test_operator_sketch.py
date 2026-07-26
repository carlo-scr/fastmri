import math

import torch

from poc_natural.sketchsense import (
    average_pool_conditioning_gram,
    sketch_identity_measurements_batched,
)
from src.samplers.operator_sketch import (
    AtomInformation,
    OperatorCovarianceSketch,
    build_atom_information,
    conditional_atom_scores,
    greedy_logdet_select,
    real_flatten,
    real_unflatten,
    sketch_operator_posterior_covariance,
)
from src.samplers.mri_forward import fft2c
from src.samplers.multicoil_pv import run_fakgd_mc_pv
from src.samplers.sense import MultiCoilSENSE


def _scaled_identity_probes(dimension: int, dtype=torch.float64):
    """Full deterministic probe set with (Omega Omega^T) / q = I."""
    return math.sqrt(dimension) * torch.eye(dimension, dtype=dtype)


def test_real_operator_sketch_recovers_exact_linear_covariance():
    dtype = torch.float64
    sigma = 0.6
    jacobian = torch.tensor(
        [
            [2.0, 0.3, 0.1],
            [0.3, 1.2, -0.2],
            [0.1, -0.2, 0.8],
        ],
        dtype=dtype,
    )
    operator = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.0, 1.5, -0.25],
            [0.3, 0.0, 0.9],
            [-0.4, 0.2, 0.7],
        ],
        dtype=dtype,
    )
    x = torch.tensor([0.2, -0.1, 0.7], dtype=dtype)

    def denoiser(value, _sigma):
        return jacobian @ value

    sketch = sketch_operator_posterior_covariance(
        x_t=x,
        mu=denoiser(x, sigma),
        sigma_t=sigma,
        denoiser=denoiser,
        forward=lambda value: operator @ value,
        adjoint=lambda value: operator.T @ value,
        measurement_template=torch.zeros(operator.shape[0], dtype=dtype),
        probes=_scaled_identity_probes(operator.shape[0], dtype=dtype),
        fd_step=1e-3,
        psd_rtol=1e-10,
        psd_atol=1e-12,
    )

    exact = operator @ (sigma**2 * jacobian) @ operator.T
    reconstructed = sketch.measurement_factor @ sketch.measurement_factor.T
    torch.testing.assert_close(
        sketch.diagonal_hutchinson_real,
        exact.diag(),
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(reconstructed, exact, rtol=1e-9, atol=1e-9)
    assert sketch.rank == 3
    assert sketch.diagnostics["core_negative_count"] == 0


def test_complex_operator_uses_stacked_real_coordinates():
    dtype = torch.float64
    complex_dtype = torch.complex128
    sigma = 0.4
    # A real-coordinate PSD denoiser Jacobian that mixes real and imaginary
    # channels and therefore cannot be treated as a scalar complex spectrum.
    base = torch.tensor(
        [
            [1.0, 0.2, -0.1, 0.3],
            [0.0, 0.8, 0.4, -0.2],
            [0.3, -0.1, 0.9, 0.1],
            [0.2, 0.2, -0.3, 1.1],
        ],
        dtype=dtype,
    )
    jacobian_real = base @ base.T + 0.2 * torch.eye(4, dtype=dtype)
    operator = torch.tensor(
        [
            [1.0 + 0.2j, -0.3 + 0.5j],
            [0.4 - 0.1j, 0.8 + 0.6j],
            [-0.2 + 0.7j, 0.5 - 0.4j],
        ],
        dtype=complex_dtype,
    )
    x = torch.tensor([0.3 + 0.4j, -0.2 + 0.1j], dtype=complex_dtype)

    def denoiser(value, _sigma):
        output_real = jacobian_real @ real_flatten(value)
        return real_unflatten(output_real, value)

    sketch = sketch_operator_posterior_covariance(
        x_t=x,
        mu=denoiser(x, sigma),
        sigma_t=sigma,
        denoiser=denoiser,
        forward=lambda value: operator @ value,
        adjoint=lambda value: operator.conj().T @ value,
        measurement_template=torch.zeros(3, dtype=complex_dtype),
        probes=_scaled_identity_probes(6, dtype=dtype),
        fd_step=2e-3,
        psd_rtol=1e-10,
        psd_atol=1e-12,
    )

    re, im = operator.real, operator.imag
    operator_real = torch.cat(
        (
            torch.cat((re, -im), dim=1),
            torch.cat((im, re), dim=1),
        ),
        dim=0,
    )
    exact = operator_real @ (sigma**2 * jacobian_real) @ operator_real.T
    exact_complex_diagonal = exact.diag()[:3] + exact.diag()[3:]
    reconstructed = sketch.measurement_factor @ sketch.measurement_factor.T

    torch.testing.assert_close(
        sketch.diagonal,
        exact_complex_diagonal,
        rtol=2e-10,
        atol=2e-10,
    )
    torch.testing.assert_close(reconstructed, exact, rtol=2e-9, atol=2e-9)
    assert sketch.rank == 4


def test_psd_projection_reports_indefinite_learned_jacobian():
    dtype = torch.float64
    jacobian = torch.diag(torch.tensor([2.0, -0.75, 0.5], dtype=dtype))
    x = torch.ones(3, dtype=dtype)

    def denoiser(value, _sigma):
        return jacobian @ value

    sketch = sketch_operator_posterior_covariance(
        x_t=x,
        mu=denoiser(x, 1.0),
        sigma_t=1.0,
        denoiser=denoiser,
        forward=lambda value: value,
        adjoint=lambda value: value,
        measurement_template=torch.zeros(3, dtype=dtype),
        probes=_scaled_identity_probes(3, dtype=dtype),
        fd_step=1e-3,
    )

    assert sketch.diagnostics["core_negative_count"] == 1
    assert sketch.diagnostics["core_negative_mass"] > 0
    assert sketch.rank == 2
    eigenvalues = torch.linalg.eigvalsh(
        sketch.measurement_factor @ sketch.measurement_factor.T
    )
    assert bool((eigenvalues >= -1e-12).all())


def _manual_redundant_sketch() -> OperatorCovarianceSketch:
    dtype = torch.float64
    factor = torch.tensor(
        [
            [3.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.5],
        ],
        dtype=dtype,
    )
    template = torch.zeros(3, dtype=dtype)
    return OperatorCovarianceSketch(
        measurement_template=template,
        image_template=template,
        omega=torch.empty(3, 0, dtype=dtype),
        response=torch.empty(3, 0, dtype=dtype),
        image_products=torch.empty(3, 0, dtype=dtype),
        measurement_factor=factor,
        image_factor=factor,
        diagonal_hutchinson_real=factor.square().sum(1),
        diagonal_low_rank_real=factor.square().sum(1),
        diagonal_residual_real=torch.zeros(3, dtype=dtype),
        core_eigenvalues=torch.ones(2, dtype=dtype),
        diagnostics={},
    )


def test_greedy_logdet_avoids_redundant_high_marginal_atom():
    sketch = _manual_redundant_sketch()
    information = build_atom_information(
        sketch,
        atom_ids=torch.tensor([0, 1, 2]),
        noise_variance=1.0,
        complex_noise_is_total=False,
    )
    top_marginal = torch.topk(information.marginal_scores, 2).indices.tolist()
    selected, gains = greedy_logdet_select(information, budget=2)

    assert set(top_marginal) == {0, 1}
    assert selected == [0, 2]
    assert gains[0] > gains[1] > 0


def test_logdet_conditions_on_already_observed_atom():
    sketch = _manual_redundant_sketch()
    information = build_atom_information(
        sketch,
        atom_ids=torch.tensor([0, 1, 2]),
        noise_variance=1.0,
        complex_noise_is_total=False,
    )

    # Atom 0 is already observed. Atom 1 points in the same uncertainty
    # direction and should lose to the slightly smaller but orthogonal atom 2.
    unconditioned, _ = greedy_logdet_select(
        information, budget=1, eligible=[1, 2]
    )
    conditioned, _ = greedy_logdet_select(
        information,
        budget=1,
        eligible=[1, 2],
        conditioned_on=[0],
    )
    marginal = conditional_atom_scores(
        information, conditioned_on=[0]
    )

    assert unconditioned == [1]
    assert conditioned == [2]
    assert marginal[2] > marginal[1]
    assert torch.isneginf(marginal[0])

    externally_conditioned, _ = greedy_logdet_select(
        information,
        budget=1,
        eligible=[1, 2],
        initial_gram=torch.tensor(
            [[9.0, 0.0], [0.0, 0.0]], dtype=torch.float64
        ),
    )
    assert externally_conditioned == [2]


def test_headline_sketch_does_not_reuse_probes_for_diagonal_completion():
    dtype = torch.float64
    direction = torch.tensor([1.0, 2.0, -0.5], dtype=dtype)
    jacobian = torch.outer(direction, direction)
    x = torch.ones(3, dtype=dtype)

    def denoiser(value, _sigma):
        return jacobian @ value

    sketch = sketch_operator_posterior_covariance(
        x_t=x,
        mu=denoiser(x, 1.0),
        sigma_t=1.0,
        denoiser=denoiser,
        forward=lambda value: value,
        adjoint=lambda value: value,
        measurement_template=torch.zeros_like(x),
        num_probes=1,
        generator=torch.Generator(device="cpu").manual_seed(41),
        fd_step=1e-3,
    )
    reconstructed = sketch.measurement_factor @ sketch.measurement_factor.T
    torch.testing.assert_close(reconstructed, jacobian, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        sketch.diagonal_residual_real,
        torch.zeros_like(sketch.diagonal_residual_real),
    )
    assert sketch.diagnostics["diagonal_completion"] == "none"


def test_complex_noise_power_is_split_between_real_components():
    dtype = torch.complex128
    template = torch.zeros(1, dtype=dtype)
    factor = torch.tensor([[1.0], [0.0]], dtype=torch.float64)
    sketch = OperatorCovarianceSketch(
        measurement_template=template,
        image_template=template,
        omega=torch.empty(2, 0, dtype=torch.float64),
        response=torch.empty(2, 0, dtype=torch.float64),
        image_products=torch.empty(2, 0, dtype=torch.float64),
        measurement_factor=factor,
        image_factor=factor,
        diagonal_hutchinson_real=torch.tensor([1.0, 0.0], dtype=torch.float64),
        diagonal_low_rank_real=torch.tensor([1.0, 0.0], dtype=torch.float64),
        diagonal_residual_real=torch.zeros(2, dtype=torch.float64),
        core_eigenvalues=torch.ones(1, dtype=torch.float64),
        diagnostics={},
    )
    info = build_atom_information(
        sketch,
        atom_ids=torch.tensor([0]),
        noise_variance=2.0,  # E|n|^2 -> variance 1 in each real component
        complex_noise_is_total=True,
    )
    torch.testing.assert_close(
        info.marginal_scores[0],
        torch.tensor(0.5 * math.log(2.0), dtype=torch.float64),
    )


def test_all_mri_operator_policies_acquire_exact_budget():
    height, width, coils = 4, 5, 2
    dtype = torch.complex64
    truth = torch.complex(
        torch.linspace(-0.5, 0.5, height * width).reshape(height, width),
        torch.zeros(height, width),
    ).to(dtype)
    sens = torch.stack(
        (
            torch.ones(height, width, dtype=dtype),
            (0.7 + 0.2j) * torch.ones(height, width, dtype=dtype),
        )
    )
    full_kspace = fft2c(sens * truth.unsqueeze(0))
    mask = torch.zeros(height, width)
    mask[:, width // 2] = 1.0
    noise_variance = torch.full((coils,), 1e-3)
    operator = MultiCoilSENSE(mask, sens, noise_variance)
    observed = mask * full_kspace
    schedule = torch.tensor([1.0, 0.3, 0.1])

    def identity_denoiser(value, _sigma):
        return value

    for policy in [
        "operator_diag", "sketch_marginal", "sketch_logdet"
    ]:
        result = run_fakgd_mc_pv(
            y_mc=observed,
            sense_op=operator,
            sigma_schedule=schedule,
            denoiser_fn=identity_denoiser,
            sigma_i_sq_init=torch.full((height, width), 1e-3),
            use_pv_gate=False,
            refine_sens=False,
            active_lines=1,
            active_rounds=1,
            active_after_frac=0.5,
            active_until_frac=0.5,
            active_score=policy,
            sketch_rank=2,
            sketch_fd_step=1e-3,
            y_mc_full=full_kspace,
            seed=9,
            return_diagnostics=True,
        )
        assert int((result["mask_final"].sum(0) > 0).sum().item()) == 2
        assert bool(torch.isfinite(result["recon"]).all())
        rows = result["diagnostics"]["operator_sketches"]
        assert len(rows) == 1
        assert rows[0]["policy"] == policy


def test_natural_identity_sketch_batches_jvps_but_factorizes_per_image():
    class LinearVPDenoiser:
        def __init__(self):
            self.abar = torch.tensor([0.8])
            self.nfe = 0

        def x0hat(self, value, _t):
            self.nfe += value.shape[0]
            # Identity in x_tilde = x_t / sqrt(abar) coordinates.
            return value / math.sqrt(0.8)

    denoiser = LinearVPDenoiser()
    xt = torch.linspace(-1, 1, 8).reshape(2, 1, 2, 2)
    sketches = sketch_identity_measurements_batched(
        denoiser,
        xt,
        0,
        num_probes=8,
        fd_step=1e-3,
        generator=torch.Generator(device="cpu").manual_seed(4),
    )

    assert len(sketches) == 2
    assert denoiser.nfe == 2 * 8 * 2
    for sketch in sketches:
        assert sketch.omega.shape == (4, 8)
        assert sketch.measurement_factor.shape[0] == 4
        assert bool((sketch.diagonal >= 0).all())


def test_average_pool_observation_contributes_external_precision_gram():
    dtype = torch.float64
    template = torch.zeros(1, 2, 2, dtype=dtype)
    factor = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=dtype,
    )
    sketch = OperatorCovarianceSketch(
        measurement_template=template,
        image_template=template,
        omega=torch.empty(4, 0, dtype=dtype),
        response=torch.empty(4, 0, dtype=dtype),
        image_products=torch.empty(4, 0, dtype=dtype),
        measurement_factor=factor,
        image_factor=factor,
        diagonal_hutchinson_real=factor.square().sum(1),
        diagonal_low_rank_real=factor.square().sum(1),
        diagonal_residual_real=torch.zeros(4, dtype=dtype),
        core_eigenvalues=torch.ones(2, dtype=dtype),
        diagnostics={},
    )

    gram = average_pool_conditioning_gram(
        sketch,
        block=2,
        noise_variance=0.5,
    )
    expected = torch.tensor(
        [[0.5, 1.0], [1.0, 2.0]], dtype=dtype
    )
    torch.testing.assert_close(gram, expected)


def test_random_probe_count_sweeps_are_nested():
    x = torch.zeros(5, dtype=torch.float64)

    def identity(value, _sigma):
        return value

    def run(probes):
        return sketch_operator_posterior_covariance(
            x_t=x,
            mu=x,
            sigma_t=1.0,
            denoiser=identity,
            forward=lambda value: value,
            adjoint=lambda value: value,
            measurement_template=torch.zeros_like(x),
            num_probes=probes,
            generator=torch.Generator(device="cpu").manual_seed(123),
        )

    two = run(2)
    four = run(4)
    torch.testing.assert_close(two.omega, four.omega[:, :2])
