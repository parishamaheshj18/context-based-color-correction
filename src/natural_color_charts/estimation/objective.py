"""Pure MAP objective for spectral illuminant estimation (PIPELINE.md, Stage 7).

minimize Σ_k w_k ‖c(ρ_k^obs) − c(ρ_k^pred(θ, b_k))‖² / σ_c² + Σ_k b_kᵀ Σ_k⁻¹ b_k + λ·prior(θ)
where c(·) is RGB chromaticity, θ the illuminant temperature in mired, and b_k reflectance
coefficients. Chromaticity removes the unknown per-patch brightness g_k.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from natural_color_charts.domain import SpectralArray
from natural_color_charts.spectral.illuminants import IlluminantFamily
from natural_color_charts.spectral.priors import ReflectancePrior
from natural_color_charts.spectral.rendering import chromaticity


@dataclass(frozen=True)
class MiredPrior:
    """Gaussian prior on the illuminant temperature from scene context."""

    center_mired: float
    sigma_mired: float
    weight: float


@dataclass(frozen=True, eq=False)
class FitProblem:
    """One illuminant family fitted to K observed patches."""

    sensitivity_3xn: SpectralArray
    family: IlluminantFamily
    observed_kx3: SpectralArray
    weights_k: SpectralArray
    priors: Sequence[ReflectancePrior]
    chromaticity_sigma: float
    mired_prior: MiredPrior | None = None

    @property
    def mired_bounds(self) -> tuple[float, float]:
        return 1e6 / self.family.cct_max_k, 1e6 / self.family.cct_min_k

    @property
    def coefficient_sizes(self) -> list[int]:
        return [prior.components for prior in self.priors]


@dataclass(frozen=True, eq=False)
class FitResult:
    """Solution of one FitProblem."""

    family: str
    mired: float
    coefficients: list[SpectralArray]
    cost: float
    jacobian: SpectralArray
    residuals: SpectralArray


def split_parameters(x: SpectralArray, sizes: Sequence[int]) -> tuple[float, list[SpectralArray]]:
    """Unpack [mired, b_1, ..., b_K]."""
    offsets = np.cumsum([1, *sizes])
    return float(x[0]), [x[offsets[i] : offsets[i + 1]] for i in range(len(sizes))]


def predicted_responses(
    problem: FitProblem, mired: float, coefficients: Sequence[SpectralArray]
) -> SpectralArray:
    """Camera RGB of each patch's modeled reflectance under the family member at mired."""
    illuminant = problem.family.spd(1e6 / mired)
    reflectances = np.stack(
        [p.mean_n + p.basis_nxk @ b for p, b in zip(problem.priors, coefficients, strict=True)]
    )
    return reflectances * illuminant @ problem.sensitivity_3xn.T


def residual_vector(x: SpectralArray, problem: FitProblem) -> SpectralArray:
    """Stacked data, reflectance-prior, and context-prior residuals in standard-deviation units."""
    mired, coefficients = split_parameters(x, problem.coefficient_sizes)
    predicted = chromaticity(predicted_responses(problem, mired, coefficients))
    observed = chromaticity(problem.observed_kx3)
    scale = np.sqrt(problem.weights_k)[:, None] / problem.chromaticity_sigma
    data = ((observed - predicted) * scale).ravel()
    reflectance = np.concatenate(
        [b / np.sqrt(p.variances_k) for p, b in zip(problem.priors, coefficients, strict=True)]
    )
    parts = [data, reflectance]
    if problem.mired_prior is not None:
        prior = problem.mired_prior
        parts.append(
            np.array([np.sqrt(prior.weight) * (mired - prior.center_mired) / prior.sigma_mired])
        )
    return np.concatenate(parts)


def initial_mired(problem: FitProblem, step_mired: float) -> float:
    """Grid search over the family's range with mean reflectances."""
    low, high = problem.mired_bounds
    candidates = np.arange(low, high + step_mired, step_mired).clip(low, high)
    zeros = np.zeros(sum(problem.coefficient_sizes))
    costs = [float(np.sum(residual_vector(np.r_[m, zeros], problem) ** 2)) for m in candidates]
    return float(candidates[int(np.argmin(costs))])


def solve_fit(
    problem: FitProblem,
    step_mired: float,
    loss: str,
    loss_scale: float,
    start: SpectralArray | None = None,
) -> FitResult:
    """Robust nonlinear least squares from a grid-search (or given) start."""
    if start is None:
        start = np.r_[initial_mired(problem, step_mired), np.zeros(sum(problem.coefficient_sizes))]
    low, high = problem.mired_bounds
    lower = np.r_[low, np.full(start.size - 1, -np.inf)]
    upper = np.r_[high, np.full(start.size - 1, np.inf)]
    start = np.clip(start, lower + 1e-9, upper - 1e-9)
    result = least_squares(
        residual_vector,
        start,
        args=(problem,),
        bounds=(lower, upper),
        loss=loss,
        f_scale=loss_scale,
    )
    mired, coefficients = split_parameters(result.x, problem.coefficient_sizes)
    return FitResult(
        family=problem.family.name,
        mired=mired,
        coefficients=coefficients,
        cost=float(2.0 * result.cost),
        jacobian=result.jac,
        residuals=result.fun,
    )


def patch_consistency_cost(
    observed_rgb: SpectralArray,
    prior: ReflectancePrior,
    sensitivity_3xn: SpectralArray,
    illuminant: SpectralArray,
    chromaticity_sigma: float,
) -> float:
    """Minimum penalized cost (data misfit + prior Mahalanobis) of one patch under an illuminant.

    Roughly χ²-distributed with 2 degrees of freedom when the patch agrees with the illuminant.
    """
    observed = chromaticity(observed_rgb)[0]

    def residuals(b: SpectralArray) -> SpectralArray:
        response = (prior.mean_n + prior.basis_nxk @ b) * illuminant @ sensitivity_3xn.T
        data = (observed - chromaticity(response)[0]) / chromaticity_sigma
        return np.concatenate([data, b / np.sqrt(prior.variances_k)])

    return float(2.0 * least_squares(residuals, np.zeros(prior.components)).cost)
