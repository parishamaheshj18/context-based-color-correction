"""Stage 7: spectral illuminant estimation from reference patches and reflectance priors."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping, Sequence

import numpy as np

from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.config import EstimationConfig
from natural_color_charts.domain import (
    NEUTRAL_CLASSES,
    CameraSensitivity,
    IlluminantEstimate,
    PatchMeasurement,
    ReferenceClass,
    SceneContext,
    normalize_white,
)
from natural_color_charts.estimation.base import (
    EstimationInput,
    EstimationOutcome,
    EstimationUnavailableError,
)
from natural_color_charts.estimation.objective import (
    FitProblem,
    FitResult,
    MiredPrior,
    patch_consistency_cost,
    predicted_responses,
    solve_fit,
)
from natural_color_charts.spectral.illuminants import IlluminantFamily
from natural_color_charts.spectral.priors import ReflectancePrior
from natural_color_charts.spectral.rendering import white_response

logger = logging.getLogger(__name__)

MIN_PRIOR_SIGMA_MIRED = 5.0


def usable_patches(patches: Sequence[PatchMeasurement]) -> list[PatchMeasurement]:
    """Patches that survived Stage 5 with a positive weight."""
    return [p for p in patches if p.kept and p.weight > 0]


def mired_prior_for(family: str, context: SceneContext, prior_weight: float) -> MiredPrior | None:
    """Context prior from the most confident hypothesis for this family, if any."""
    hypotheses = [
        h for h in context.illuminant_hypotheses if h.family == family and h.confidence > 0
    ]
    if not hypotheses:
        return None
    best = max(hypotheses, key=lambda h: h.confidence)
    low, high = sorted(1e6 / np.asarray(best.cct_range_k, dtype=float))
    return MiredPrior(
        center_mired=float((low + high) / 2),
        sigma_mired=max(float((high - low) / 2), MIN_PRIOR_SIGMA_MIRED),
        weight=prior_weight * best.confidence,
    )


class SpectralIlluminantEstimator:
    """Fits an illuminant family to reference patches under their reflectance priors."""

    name = "spectral"

    def __init__(
        self,
        families: Mapping[str, IlluminantFamily],
        priors: Mapping[ReferenceClass, ReflectancePrior],
        config: EstimationConfig,
    ) -> None:
        self._families = dict(families)
        self._priors = dict(priors)
        self._config = config

    def estimate(self, request: EstimationInput) -> EstimationOutcome:
        if request.sensitivity is None:
            raise EstimationUnavailableError("no camera sensitivity data")
        patches = [p for p in usable_patches(request.patches) if p.label in self._priors]
        if not patches:
            raise EstimationUnavailableError("no usable reference patches")
        families = self._families_for(request.context)
        fit, problem = self._fit_best(families, patches, request)
        outliers = self._find_outliers(fit, problem, patches, request)
        if outliers:
            patches = [p for i, p in enumerate(patches) if i not in outliers]
            fit, problem = self._fit_best(families, patches, request)
        return EstimationOutcome(
            estimate=self._to_estimate(fit, problem, patches, request.sensitivity),
            patches=self._mark_outliers(request.patches, patches),
        )

    def _prior_for(self, patch: PatchMeasurement) -> ReflectancePrior:
        prior = self._priors[patch.label]
        if patch.known_reflectance is not None:
            return dataclasses.replace(prior, mean_n=np.asarray(patch.known_reflectance))
        return prior

    def _families_for(self, context: SceneContext) -> list[IlluminantFamily]:
        named = [h.family for h in context.illuminant_hypotheses if h.family in self._families]
        return [self._families[n] for n in dict.fromkeys(named)] or list(self._families.values())

    def _problem(
        self,
        family: IlluminantFamily,
        patches: Sequence[PatchMeasurement],
        request: EstimationInput,
    ) -> FitProblem:
        assert request.sensitivity is not None
        return FitProblem(
            sensitivity_3xn=request.sensitivity.sensitivity_3xn,
            family=family,
            observed_kx3=np.stack([p.rgb for p in patches]),
            weights_k=np.array([p.weight for p in patches]),
            priors=[self._prior_for(p) for p in patches],
            chromaticity_sigma=self._config.chromaticity_sigma,
            mired_prior=mired_prior_for(
                family.name, request.context, self._config.context_prior_weight
            ),
        )

    def _solve(self, problem: FitProblem, start: np.ndarray | None = None) -> FitResult:
        config = self._config
        return solve_fit(
            problem, config.grid_step_mired, config.robust_loss, config.robust_scale, start
        )

    def _fit_best(
        self,
        families: Sequence[IlluminantFamily],
        patches: Sequence[PatchMeasurement],
        request: EstimationInput,
    ) -> tuple[FitResult, FitProblem]:
        candidates = []
        for family in families:
            problem = self._problem(family, patches, request)
            candidates.append((self._solve(problem), problem))
        return min(candidates, key=lambda pair: pair[0].cost)

    def _find_outliers(
        self,
        fit: FitResult,
        problem: FitProblem,
        patches: Sequence[PatchMeasurement],
        request: EstimationInput,
    ) -> set[int]:
        """Leave-one-out: flag patches inconsistent with the illuminant fitted to the others.

        Measuring each patch against the others' fit matters because the robust loss already
        down-weights an outlier in the joint fit, hiding how much it disagrees.
        """
        if len(patches) < self._config.loo_min_references:
            return set()
        flagged: list[tuple[float, int]] = []
        for index in range(len(patches)):
            if patches[index].source == "user":
                continue  # the user's matches are ground truth, never outliers
            others = [p for i, p in enumerate(patches) if i != index]
            loo_problem = self._problem(problem.family, others, request)
            start = np.concatenate(
                [[fit.mired], *[b for i, b in enumerate(fit.coefficients) if i != index]]
            )
            loo_fit = self._solve(loo_problem, start)
            cost = patch_consistency_cost(
                patches[index].rgb,
                self._prior_for(patches[index]),
                problem.sensitivity_3xn,
                problem.family.spd(1e6 / loo_fit.mired),
                self._config.chromaticity_sigma,
            )
            if cost > self._config.loo_outlier_cost:
                flagged.append((cost, index))
        allowed = max(0, len(patches) - 2)
        return {index for _, index in sorted(flagged, reverse=True)[:allowed]}

    def _to_estimate(
        self,
        fit: FitResult,
        problem: FitProblem,
        patches: Sequence[PatchMeasurement],
        sensitivity: CameraSensitivity,
    ) -> IlluminantEstimate:
        spectrum = problem.family.spd(1e6 / fit.mired)
        white = white_response(sensitivity.sensitivity_3xn, spectrum)
        mired_sigma = self._mired_sigma(fit)
        shifted = problem.family.spd(1e6 / min(fit.mired + mired_sigma, problem.mired_bounds[1]))
        residual_deg = self._mean_patch_error(fit, problem)
        support = 1.0 - np.exp(-float(problem.weights_k.sum()))
        confidence = float(
            np.exp(-residual_deg / self._config.confidence_residual_scale_deg) * support
        )
        return IlluminantEstimate(
            white_rgb=normalize_white(white),
            method=self.name,
            confidence=confidence,
            family=fit.family,
            cct_k=1e6 / fit.mired,
            spectrum=spectrum,
            cct_uncertainty_k=1e6 * mired_sigma / fit.mired**2,
            angular_uncertainty_deg=angular_error_deg(
                white_response(sensitivity.sensitivity_3xn, shifted), white
            ),
            used_labels=tuple(p.label.value for p in patches),
            notes=(f"mean patch residual {residual_deg:.2f}°",),
        )

    @staticmethod
    def _mired_sigma(fit: FitResult) -> float:
        jacobian = fit.jacobian
        dof = max(1, jacobian.shape[0] - jacobian.shape[1])
        scale = max(1.0, float(fit.residuals @ fit.residuals) / dof)
        try:
            covariance = np.linalg.inv(jacobian.T @ jacobian) * scale
        except np.linalg.LinAlgError:
            return float("nan")
        return float(np.sqrt(max(covariance[0, 0], 0.0)))

    @staticmethod
    def _mean_patch_error(fit: FitResult, problem: FitProblem) -> float:
        predicted = predicted_responses(problem, fit.mired, fit.coefficients)
        errors = [
            angular_error_deg(p, o) for p, o in zip(predicted, problem.observed_kx3, strict=True)
        ]
        return float(np.average(errors, weights=problem.weights_k))

    @staticmethod
    def _mark_outliers(
        all_patches: Sequence[PatchMeasurement], used: Sequence[PatchMeasurement]
    ) -> tuple[PatchMeasurement, ...]:
        used_ids = {id(p) for p in used}
        return tuple(
            dataclasses.replace(
                p, kept=False, rejection_reason="inconsistent with other references"
            )
            if p.kept and p.weight > 0 and id(p) not in used_ids
            else p
            for p in all_patches
        )


class NeutralPatchEstimator:
    """White point straight from near-neutral patches; needs no spectral data."""

    name = "neutral_patches"

    def __init__(self, confidence_scale: float) -> None:
        self._confidence_scale = confidence_scale

    def estimate(self, request: EstimationInput) -> EstimationOutcome:
        neutrals = [
            p
            for p in usable_patches(request.patches)
            if p.label in NEUTRAL_CLASSES or p.is_neutral_known_color
        ]
        if not neutrals:
            raise EstimationUnavailableError("no usable neutral patches")
        chromas = np.stack([p.rgb / p.rgb.sum() for p in neutrals])
        weights = np.array([p.weight for p in neutrals])
        white = normalize_white(np.average(chromas, axis=0, weights=weights))
        spread = float(np.average([angular_error_deg(c, white) for c in chromas], weights=weights))
        support = 1.0 - np.exp(-float(weights.sum()))
        return EstimationOutcome(
            IlluminantEstimate(
                white_rgb=white,
                method=self.name,
                confidence=float(np.exp(-spread / self._confidence_scale) * support),
                used_labels=tuple(p.label.value for p in neutrals),
            )
        )
