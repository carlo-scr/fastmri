"""Real-embedded operator-space covariance sketches for diffusion denoisers.

The local Tweedie approximation writes

    Sigma_t = sigma_t**2 * d denoiser(x_t, sigma_t) / d x_t.

For a complete bank of candidate measurements ``B``, SketchSense needs

    C_t = B Sigma_t B^*.

This module never forms either Jacobian or covariance. It applies ``C_t`` to
a small real probe matrix with central finite-difference JVPs, constructs a
positive-semidefinite Nyström factor, and exposes grouped log-determinant
design utilities.

All linear algebra is performed in the real embedding. A complex tensor is
flattened as ``[real.flatten(), imag.flatten()]``. Consequently the code is
valid for denoisers that mix real and imaginary channels; it does not assume a
complex-linear Jacobian. The supplied ``adjoint`` must be the true Hilbert
adjoint of the *unmasked, unnormalized* candidate bank under the real inner
product. In multi-coil MRI, do not pass the normalized, mask-restricted SENSE
pseudoinverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


TensorFn = Callable[[torch.Tensor], torch.Tensor]
DenoiserFn = Callable[[torch.Tensor, float], torch.Tensor]


def real_flatten(value: torch.Tensor) -> torch.Tensor:
    """Flatten a tensor in the network's real coordinate representation."""
    if torch.is_complex(value):
        return torch.cat((value.real.reshape(-1), value.imag.reshape(-1)))
    return value.reshape(-1)


def real_unflatten(vector: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`real_flatten`, using ``template`` for shape/dtype."""
    n = template.numel()
    if torch.is_complex(template):
        if vector.numel() != 2 * n:
            raise ValueError(
                f"expected {2 * n} real coordinates, got {vector.numel()}"
            )
        real = vector[:n].reshape(template.shape).to(template.real.dtype)
        imag = vector[n:].reshape(template.shape).to(template.real.dtype)
        return torch.complex(real, imag).to(template.dtype)
    if vector.numel() != n:
        raise ValueError(f"expected {n} coordinates, got {vector.numel()}")
    return vector.reshape(template.shape).to(template.dtype)


def collapse_real_diagonal(
    diagonal_real: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    """Return measurement-shaped marginal variances.

    For a complex measurement, each output entry is
    ``Var(real) + Var(imag)``. For a real measurement, this is just a reshape.
    """
    n = template.numel()
    if torch.is_complex(template):
        if diagonal_real.numel() != 2 * n:
            raise ValueError("complex diagonal has the wrong real dimension")
        return (
            diagonal_real[:n] + diagonal_real[n:]
        ).reshape(template.shape)
    if diagonal_real.numel() != n:
        raise ValueError("real diagonal has the wrong dimension")
    return diagonal_real.reshape(template.shape)


def expand_measurement_values_to_real(
    values: float | torch.Tensor,
    template: torch.Tensor,
    *,
    complex_values_are_total: bool,
) -> torch.Tensor:
    """Broadcast measurement-shaped values into the real embedding.

    ``complex_values_are_total=True`` interprets each complex value as total
    power, e.g. ``E|n_i|^2``. A proper complex noise model then assigns half to
    each real component. Set it to ``False`` when values already denote
    per-real-component variances.
    """
    real_dtype = template.real.dtype if torch.is_complex(template) else template.dtype
    value = torch.as_tensor(values, device=template.device, dtype=real_dtype)
    value = torch.broadcast_to(value, template.shape).reshape(-1)
    if torch.is_complex(template):
        if complex_values_are_total:
            value = value / 2.0
        return torch.cat((value, value))
    return value


def expand_atom_ids_to_real(
    atom_ids: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    """Flatten measurement-shaped atom IDs using the same real embedding."""
    ids = torch.broadcast_to(atom_ids.to(template.device), template.shape).reshape(-1)
    ids = ids.to(torch.long)
    return torch.cat((ids, ids)) if torch.is_complex(template) else ids


@dataclass
class OperatorCovarianceSketch:
    """Products and PSD factors for ``B Sigma_t B^*`` in real coordinates."""

    measurement_template: torch.Tensor
    image_template: torch.Tensor
    omega: torch.Tensor                 # (measurement_real_dim, q)
    response: torch.Tensor              # C @ omega, same shape
    image_products: torch.Tensor        # Sigma B^* omega, (image_real_dim, q)
    measurement_factor: torch.Tensor    # L_y, C ~= L_y L_y^T + diag(residual)
    image_factor: torch.Tensor          # L_x, cross-cov ~= L_x L_y^T
    diagonal_hutchinson_real: torch.Tensor
    diagonal_low_rank_real: torch.Tensor
    diagonal_residual_real: torch.Tensor
    core_eigenvalues: torch.Tensor
    diagnostics: dict[str, float | int | str]

    @property
    def rank(self) -> int:
        return int(self.measurement_factor.shape[1])

    @property
    def diagonal_real(self) -> torch.Tensor:
        return self.diagonal_low_rank_real + self.diagonal_residual_real

    @property
    def diagonal(self) -> torch.Tensor:
        return collapse_real_diagonal(
            self.diagonal_real, self.measurement_template
        )

    @property
    def hutchinson_diagonal(self) -> torch.Tensor:
        return collapse_real_diagonal(
            self.diagonal_hutchinson_real, self.measurement_template
        )


def _rademacher_matrix(
    rows: int,
    cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    # Generate on CPU so one counter-based CPU generator is portable across
    # CPU, CUDA, and MPS.
    # Probe-major generation makes q-probe sweeps nested: with the same seed,
    # q=4 is exactly the first four columns of q=8.
    probe = torch.randint(
        0, 2, (cols, rows), generator=generator, device="cpu"
    ).T.contiguous()
    return probe.to(device=device, dtype=dtype).mul_(2).sub_(1)


def _real_rms(value: torch.Tensor) -> torch.Tensor:
    return real_flatten(value).square().mean().sqrt()


def _psd_factors_from_products(
    omega: torch.Tensor,
    response: torch.Tensor,
    image_products: torch.Tensor,
    *,
    psd_rtol: float,
    psd_atol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Nyström factors with explicit symmetry and PSD protection."""
    core_raw = omega.T @ response
    denom = core_raw.norm().clamp_min(torch.finfo(core_raw.dtype).eps)
    asymmetry = float(((core_raw - core_raw.T).norm() / denom).item())
    core = 0.5 * (core_raw + core_raw.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(core)

    scale = eigenvalues.abs().max().clamp_min(
        torch.as_tensor(psd_atol, device=eigenvalues.device, dtype=eigenvalues.dtype)
    )
    threshold = max(float(psd_atol), float(psd_rtol) * float(scale.item()))
    abs_sum = eigenvalues.abs().sum().clamp_min(
        torch.finfo(eigenvalues.dtype).eps
    )
    negative = eigenvalues[eigenvalues < -threshold]
    negative_mass = float((-negative.sum() / abs_sum).item())
    keep = eigenvalues > threshold

    if bool(keep.any()):
        kept_values = eigenvalues[keep]
        kept_vectors = eigenvectors[:, keep]
        inverse_sqrt = kept_values.rsqrt()
        measurement_factor = (
            response @ kept_vectors
        ) * inverse_sqrt.unsqueeze(0)
        image_factor = (
            image_products @ kept_vectors
        ) * inverse_sqrt.unsqueeze(0)
    else:
        kept_values = eigenvalues.new_empty((0,))
        measurement_factor = response.new_zeros((response.shape[0], 0))
        image_factor = image_products.new_zeros((image_products.shape[0], 0))

    diagnostics: dict[str, float | int | str] = {
        "core_asymmetry": asymmetry,
        "core_min_eigenvalue": float(eigenvalues.min().item()),
        "core_max_eigenvalue": float(eigenvalues.max().item()),
        "core_negative_count": int((eigenvalues < -threshold).sum().item()),
        "core_negative_mass": negative_mass,
        "effective_rank": int(kept_values.numel()),
        "psd_threshold": threshold,
    }
    return measurement_factor, image_factor, eigenvalues, diagnostics


def operator_sketch_from_products(
    *,
    measurement_template: torch.Tensor,
    image_template: torch.Tensor,
    omega: torch.Tensor,
    response: torch.Tensor,
    image_products: torch.Tensor,
    psd_rtol: float = 1e-6,
    psd_atol: float = 1e-10,
    diagonal_completion: str = "none",
    diagnostics: dict[str, float | int | str] | None = None,
) -> OperatorCovarianceSketch:
    """Construct a protected sketch from already-computed operator products.

    This is useful when several independent problems are evaluated in one
    batched denoiser call. Columns must still be in the real embedding.
    """
    if omega.dim() != 2 or response.shape != omega.shape:
        raise ValueError("omega and response must have the same 2-D shape")
    if image_products.dim() != 2 or image_products.shape[1] != omega.shape[1]:
        raise ValueError("image_products must share omega's probe dimension")
    if omega.shape[0] != real_flatten(measurement_template).numel():
        raise ValueError("measurement products do not match their template")
    if image_products.shape[0] != real_flatten(image_template).numel():
        raise ValueError("image products do not match their template")

    measurement_factor, image_factor, core_eigenvalues, core_diagnostics = (
        _psd_factors_from_products(
            omega,
            response,
            image_products,
            psd_rtol=psd_rtol,
            psd_atol=psd_atol,
        )
    )
    diagonal_hutchinson = (omega * response).mean(dim=1)
    diagonal_low_rank = measurement_factor.square().sum(dim=1)
    if diagonal_completion == "none":
        # Reusing the Nyström probes to estimate a residual diagonal is not an
        # independent Hutchinson estimate. The headline method therefore uses
        # the pure PSD low-rank factor; the cross-Hutchinson diagonal remains a
        # separate controlled ablation.
        diagonal_residual = torch.zeros_like(diagonal_low_rank)
    elif diagonal_completion == "same_probe_heuristic":
        diagonal_residual = (
            diagonal_hutchinson - diagonal_low_rank
        ).clamp_min(0.0)
    else:
        raise ValueError(
            "diagonal_completion must be 'none' or "
            "'same_probe_heuristic'"
        )
    total_hutch_power = diagonal_hutchinson.abs().sum().clamp_min(
        torch.finfo(diagonal_hutchinson.dtype).eps
    )
    positive_hutch_power = diagonal_hutchinson.clamp_min(0).sum().clamp_min(
        torch.finfo(diagonal_hutchinson.dtype).eps
    )
    core_diagnostics.update({
        "num_probes": int(omega.shape[1]),
        "hutchinson_negative_fraction": float(
            (diagonal_hutchinson < 0).float().mean().item()
        ),
        "hutchinson_negative_mass": float(
            (
                -diagonal_hutchinson.clamp_max(0).sum()
                / total_hutch_power
            ).item()
        ),
        "low_rank_diagonal_fraction": float(
            (diagonal_low_rank.sum() / positive_hutch_power).item()
        ),
        "diagonal_completion": diagonal_completion,
    })
    if diagnostics:
        core_diagnostics.update(diagnostics)
    return OperatorCovarianceSketch(
        measurement_template=measurement_template,
        image_template=image_template,
        omega=omega,
        response=response,
        image_products=image_products,
        measurement_factor=measurement_factor,
        image_factor=image_factor,
        diagonal_hutchinson_real=diagonal_hutchinson,
        diagonal_low_rank_real=diagonal_low_rank,
        diagonal_residual_real=diagonal_residual,
        core_eigenvalues=core_eigenvalues,
        diagnostics=core_diagnostics,
    )


@torch.no_grad()
def sketch_operator_posterior_covariance(
    x_t: torch.Tensor,
    mu: torch.Tensor,
    sigma_t: float,
    denoiser: DenoiserFn,
    forward: TensorFn,
    adjoint: TensorFn,
    measurement_template: torch.Tensor,
    *,
    num_probes: int = 4,
    fd_step: float = 5e-2,
    centered: bool = True,
    probes: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    psd_rtol: float = 1e-6,
    psd_atol: float = 1e-10,
) -> OperatorCovarianceSketch:
    """Sketch ``B (sigma_t^2 J_t) B^*`` with finite-difference JVPs.

    ``probes``, when provided, must have shape ``(measurement_real_dim, q)``
    in the real embedding. Otherwise independent Rademacher probes are drawn.
    ``fd_step`` is a fraction of ``max(abs(sigma_t), 1e-8)`` after each
    pullback has been normalized to unit real-coordinate RMS.
    """
    if num_probes <= 0:
        raise ValueError("num_probes must be positive")
    if fd_step <= 0:
        raise ValueError("fd_step must be positive")
    if x_t.shape != mu.shape:
        raise ValueError("x_t and mu must have identical shapes")

    measurement_dim = real_flatten(measurement_template).numel()
    image_dim = real_flatten(x_t).numel()
    real_dtype = (
        measurement_template.real.dtype
        if torch.is_complex(measurement_template)
        else measurement_template.dtype
    )

    if probes is None:
        omega = _rademacher_matrix(
            measurement_dim,
            num_probes,
            device=measurement_template.device,
            dtype=real_dtype,
            generator=generator,
        )
    else:
        omega = probes.to(device=measurement_template.device, dtype=real_dtype)
        if omega.dim() != 2 or omega.shape[0] != measurement_dim:
            raise ValueError(
                "probes must have shape (measurement_real_dim, num_probes)"
            )
        num_probes = int(omega.shape[1])

    response = torch.zeros(
        (measurement_dim, num_probes),
        device=measurement_template.device,
        dtype=real_dtype,
    )
    image_products = torch.zeros(
        (image_dim, num_probes),
        device=x_t.device,
        dtype=x_t.real.dtype if torch.is_complex(x_t) else x_t.dtype,
    )
    delta = max(float(fd_step) * max(abs(float(sigma_t)), 1e-8), 1e-8)
    zero_pullbacks = 0

    for column in range(num_probes):
        measurement_probe = real_unflatten(
            omega[:, column], measurement_template
        )
        pullback = adjoint(measurement_probe)
        if pullback.shape != x_t.shape:
            raise ValueError(
                f"adjoint returned {tuple(pullback.shape)}, "
                f"expected {tuple(x_t.shape)}"
            )
        pullback_rms = _real_rms(pullback)
        if float(pullback_rms.item()) <= 1e-12:
            zero_pullbacks += 1
            continue
        direction = pullback / pullback_rms
        if centered:
            mu_plus = denoiser(x_t + delta * direction, sigma_t)
            mu_minus = denoiser(x_t - delta * direction, sigma_t)
            jvp_direction = (mu_plus - mu_minus) / (2.0 * delta)
        else:
            mu_plus = denoiser(x_t + delta * direction, sigma_t)
            jvp_direction = (mu_plus - mu) / delta

        # Undo pullback normalization and apply the Tweedie sigma_t^2 scale.
        image_product = (
            (float(sigma_t) ** 2) * pullback_rms * jvp_direction
        )
        measurement_product = forward(image_product)
        if measurement_product.shape != measurement_template.shape:
            raise ValueError(
                f"forward returned {tuple(measurement_product.shape)}, "
                f"expected {tuple(measurement_template.shape)}"
            )
        image_products[:, column] = real_flatten(image_product)
        response[:, column] = real_flatten(measurement_product)

    return operator_sketch_from_products(
        measurement_template=measurement_template,
        image_template=x_t,
        omega=omega,
        response=response,
        image_products=image_products,
        psd_rtol=psd_rtol,
        psd_atol=psd_atol,
        diagnostics={"zero_pullbacks": int(zero_pullbacks)},
    )


@dataclass
class AtomInformation:
    """Grouped determinant terms for a diagonal-plus-low-rank covariance."""

    grams: torch.Tensor               # (num_atoms, rank, rank)
    residual_logdet: torch.Tensor     # (num_atoms,), without the 1/2 factor
    marginal_scores: torch.Tensor     # (num_atoms,), includes the 1/2 factor
    present: torch.Tensor             # (num_atoms,) bool

    @property
    def num_atoms(self) -> int:
        return int(self.grams.shape[0])


def _resolve_atom_list(
    information: AtomInformation,
    atoms: Sequence[int] | torch.Tensor | None,
) -> list[int]:
    if atoms is None:
        return []
    if isinstance(atoms, torch.Tensor):
        if atoms.dtype == torch.bool:
            if atoms.numel() != information.num_atoms:
                raise ValueError("atom boolean mask has the wrong length")
            values = torch.nonzero(atoms, as_tuple=False).flatten().tolist()
        else:
            values = atoms.flatten().tolist()
    else:
        values = list(atoms)
    return sorted({
        int(atom) for atom in values
        if 0 <= int(atom) < information.num_atoms
        and bool(information.present[int(atom)])
    })


def _conditioned_low_rank_state(
    information: AtomInformation,
    conditioned_on: Sequence[int] | torch.Tensor | None,
    initial_gram: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    conditioned = _resolve_atom_list(information, conditioned_on)
    rank = int(information.grams.shape[-1])
    accumulated = torch.eye(
        rank,
        device=information.grams.device,
        dtype=information.grams.dtype,
    )
    if initial_gram is not None:
        if initial_gram.shape != accumulated.shape:
            raise ValueError(
                "initial_gram must have shape (sketch_rank, sketch_rank)"
            )
        accumulated = accumulated + 0.5 * (
            initial_gram.to(accumulated) + initial_gram.to(accumulated).T
        )
    for atom in conditioned:
        accumulated = accumulated + information.grams[atom]
    if rank:
        sign, current_logdet = torch.linalg.slogdet(accumulated)
        if float(sign.item()) <= 0:
            raise RuntimeError(
                "conditioned PSD atom grams produced nonpositive determinant"
            )
    else:
        current_logdet = information.grams.new_tensor(0.0)
    return accumulated, current_logdet, conditioned


def build_atom_information(
    sketch: OperatorCovarianceSketch,
    atom_ids: torch.Tensor,
    noise_variance: float | torch.Tensor,
    *,
    num_atoms: int | None = None,
    complex_noise_is_total: bool = True,
    eps: float = 1e-12,
) -> AtomInformation:
    """Group a sketch into candidate atoms for local Gaussian information gain.

    For ``C = diag(d) + L L^T`` and real-coordinate noise ``R``,

    ``log det(I + R^-1 C)`` equals

    ``sum_i log(1 + d_i/R_i) + log det(I + L^T(R+D)^-1 L)``.

    The returned terms make this identity additive across disjoint atoms.
    Complex ``noise_variance`` defaults to total proper-complex power
    ``E|n_i|^2`` and is divided equally between real and imaginary components.
    """
    ids = expand_atom_ids_to_real(atom_ids, sketch.measurement_template)
    if ids.numel() != sketch.measurement_factor.shape[0]:
        raise ValueError("atom IDs do not match measurement dimension")
    if bool((ids < -1).any()):
        raise ValueError("atom IDs must be -1 (ignored) or nonnegative")
    valid = ids >= 0
    inferred = int(ids[valid].max().item()) + 1 if bool(valid.any()) else 0
    if num_atoms is None:
        num_atoms = inferred
    if int(num_atoms) < inferred:
        raise ValueError("num_atoms is smaller than the largest atom ID")

    noise_real = expand_measurement_values_to_real(
        noise_variance,
        sketch.measurement_template,
        complex_values_are_total=complex_noise_is_total,
    ).clamp_min(eps)
    residual = sketch.diagonal_residual_real.clamp_min(0.0)
    rank = sketch.rank
    grams = sketch.measurement_factor.new_zeros(
        (int(num_atoms), rank, rank)
    )
    residual_logdet = sketch.measurement_factor.new_zeros((int(num_atoms),))
    present = torch.zeros(
        int(num_atoms), device=sketch.measurement_factor.device, dtype=torch.bool
    )

    for atom in range(int(num_atoms)):
        index = ids == atom
        if not bool(index.any()):
            continue
        present[atom] = True
        atom_noise = noise_real[index]
        atom_residual = residual[index]
        residual_logdet[atom] = torch.log1p(
            atom_residual / atom_noise
        ).sum()
        if rank:
            rows = sketch.measurement_factor[index]
            weights = (atom_noise + atom_residual).reciprocal()
            grams[atom] = rows.T @ (weights.unsqueeze(1) * rows)

    eye = torch.eye(
        rank,
        device=sketch.measurement_factor.device,
        dtype=sketch.measurement_factor.dtype,
    )
    marginal = residual_logdet.clone()
    if rank:
        for atom in range(int(num_atoms)):
            if present[atom]:
                sign, logabs = torch.linalg.slogdet(eye + grams[atom])
                if float(sign.item()) <= 0:
                    raise RuntimeError("PSD atom gram produced nonpositive determinant")
                marginal[atom] = marginal[atom] + logabs
    marginal = 0.5 * marginal
    marginal = marginal.masked_fill(~present, float("-inf"))
    return AtomInformation(
        grams=grams,
        residual_logdet=residual_logdet,
        marginal_scores=marginal,
        present=present,
    )


def diagonal_atom_scores(
    sketch: OperatorCovarianceSketch,
    atom_ids: torch.Tensor,
    noise_variance: float | torch.Tensor,
    *,
    num_atoms: int | None = None,
    complex_noise_is_total: bool = True,
    source: str = "psd",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Noise-normalized marginal diagonal scores for a controlled ablation.

    ``source="hutchinson"`` keeps signed coordinate estimates through atom
    aggregation and then applies one floor. ``source="psd"`` uses the
    diagonal of the protected diagonal-plus-low-rank sketch.
    """
    ids = expand_atom_ids_to_real(atom_ids, sketch.measurement_template)
    noise_real = expand_measurement_values_to_real(
        noise_variance,
        sketch.measurement_template,
        complex_values_are_total=complex_noise_is_total,
    ).clamp_min(eps)
    valid = ids >= 0
    inferred = int(ids[valid].max().item()) + 1 if bool(valid.any()) else 0
    if num_atoms is None:
        num_atoms = inferred
    if source == "hutchinson":
        coordinate_diagonal = sketch.diagonal_hutchinson_real
    elif source == "psd":
        coordinate_diagonal = sketch.diagonal_real
    else:
        raise ValueError("source must be 'hutchinson' or 'psd'")
    scores = coordinate_diagonal.new_zeros((int(num_atoms),))
    scores.scatter_add_(
        0,
        ids[valid],
        (coordinate_diagonal[valid] / noise_real[valid]),
    )
    return scores.clamp_min(0.0)


def conditional_atom_scores(
    information: AtomInformation,
    *,
    conditioned_on: Sequence[int] | torch.Tensor | None = None,
    initial_gram: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score each atom's conditional gain without within-batch updates.

    This is the marginal counterpart to :func:`greedy_logdet_select`. Both
    condition on the already acquired set; only the joint policy updates the
    low-rank state after each new atom in the prospective batch.
    """
    accumulated, current_logdet, conditioned = _conditioned_low_rank_state(
        information, conditioned_on, initial_gram
    )
    conditioned_set = set(conditioned)
    scores = information.marginal_scores.new_full(
        (information.num_atoms,), float("-inf")
    )
    rank = int(information.grams.shape[-1])
    for atom in range(information.num_atoms):
        if not bool(information.present[atom]) or atom in conditioned_set:
            continue
        if rank:
            sign, proposed_logdet = torch.linalg.slogdet(
                accumulated + information.grams[atom]
            )
            if float(sign.item()) <= 0:
                continue
            low_rank_gain = proposed_logdet - current_logdet
        else:
            low_rank_gain = current_logdet.new_tensor(0.0)
        scores[atom] = 0.5 * (
            information.residual_logdet[atom] + low_rank_gain
        )
    return scores


def greedy_logdet_select(
    information: AtomInformation,
    budget: int,
    *,
    eligible: Sequence[int] | torch.Tensor | None = None,
    conditioned_on: Sequence[int] | torch.Tensor | None = None,
    initial_gram: torch.Tensor | None = None,
) -> tuple[list[int], list[float]]:
    """Greedily maximize information gain conditional on acquired atoms."""
    if budget <= 0:
        return [], []
    if eligible is None:
        candidates = torch.nonzero(
            information.present, as_tuple=False
        ).flatten().tolist()
    elif isinstance(eligible, torch.Tensor):
        if eligible.dtype == torch.bool:
            if eligible.numel() != information.num_atoms:
                raise ValueError("eligible boolean mask has the wrong length")
            candidates = torch.nonzero(
                eligible & information.present, as_tuple=False
            ).flatten().tolist()
        else:
            candidates = [int(x) for x in eligible.flatten().tolist()]
    else:
        candidates = [int(x) for x in eligible]
    candidates = sorted({
        atom for atom in candidates
        if 0 <= atom < information.num_atoms
        and bool(information.present[atom])
    })

    accumulated, current_logdet, conditioned = _conditioned_low_rank_state(
        information, conditioned_on, initial_gram
    )
    conditioned_set = set(conditioned)
    candidates = [atom for atom in candidates if atom not in conditioned_set]
    rank = int(information.grams.shape[-1])
    selected: list[int] = []
    selected_set: set[int] = set()
    gains: list[float] = []

    for _ in range(min(int(budget), len(candidates))):
        best_atom = None
        best_gain = float("-inf")
        best_logdet = None
        for atom in candidates:
            if atom in selected_set:
                continue
            if rank:
                sign, proposed_logdet = torch.linalg.slogdet(
                    accumulated + information.grams[atom]
                )
                if float(sign.item()) <= 0:
                    continue
                delta_low_rank = proposed_logdet - current_logdet
            else:
                proposed_logdet = current_logdet
                delta_low_rank = current_logdet.new_tensor(0.0)
            gain = 0.5 * (
                information.residual_logdet[atom] + delta_low_rank
            )
            gain_value = float(gain.item())
            if (
                gain_value > best_gain
                or (
                    gain_value == best_gain
                    and (best_atom is None or atom < best_atom)
                )
            ):
                best_atom = atom
                best_gain = gain_value
                best_logdet = proposed_logdet
        if best_atom is None:
            break
        selected.append(best_atom)
        selected_set.add(best_atom)
        gains.append(best_gain)
        if rank:
            accumulated = accumulated + information.grams[best_atom]
            current_logdet = best_logdet
    return selected, gains
