"""Classic statistical white-balance estimators used as baselines and fallbacks."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from natural_color_charts.domain import IlluminantEstimate, LinearImage, normalize_white
from natural_color_charts.estimation.base import (
    EstimationInput,
    EstimationOutcome,
    EstimationUnavailableError,
)


def sample_valid_pixels(image: LinearImage, max_pixels: int) -> NDArray[np.float64]:
    """Return up to max_pixels unsaturated, non-black pixels as an N×3 array."""
    valid = image.valid_mask & (image.rgb.max(axis=2) > 0)
    pixels = image.rgb[valid].astype(np.float64)
    if pixels.shape[0] == 0:
        raise EstimationUnavailableError("no valid pixels")
    stride = max(1, pixels.shape[0] // max_pixels)
    return pixels[::stride]


def minkowski_white(pixels: NDArray[np.float64], power: float) -> NDArray[np.float64]:
    """Shades-of-gray white point: the per-channel Minkowski p-norm mean (Finlayson 2004)."""
    return normalize_white(np.mean(pixels**power, axis=0) ** (1.0 / power))


def percentile_white(pixels: NDArray[np.float64], percentile: float) -> NDArray[np.float64]:
    """Robust white-patch estimate: a high per-channel percentile instead of the maximum."""
    return normalize_white(np.percentile(pixels, percentile, axis=0))


def gradient_magnitudes(image: LinearImage, sigma: float, max_pixels: int) -> NDArray[np.float64]:
    """Per-channel gradient magnitude of a Gaussian-smoothed image at valid pixels."""
    smoothed = cv2.GaussianBlur(image.rgb, (0, 0), sigmaX=sigma) if sigma > 0 else image.rgb
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edges = magnitude[image.valid_mask].astype(np.float64)
    edges = edges[edges.max(axis=1) > 0]
    if edges.shape[0] == 0:
        raise EstimationUnavailableError("no edges")
    return edges[:: max(1, edges.shape[0] // max_pixels)]


class _StatisticalEstimator:
    name = "statistical"

    def __init__(self, confidence: float, max_pixels: int) -> None:
        self._confidence = confidence
        self._max_pixels = max_pixels

    def estimate(self, request: EstimationInput) -> EstimationOutcome:
        white = self._white(request.image)
        return EstimationOutcome(
            IlluminantEstimate(white_rgb=white, method=self.name, confidence=self._confidence)
        )

    def _white(self, image: LinearImage) -> NDArray[np.float64]:
        raise NotImplementedError


class GrayWorldEstimator(_StatisticalEstimator):
    """Assumes the average scene reflectance is achromatic."""

    name = "gray_world"

    def _white(self, image: LinearImage) -> NDArray[np.float64]:
        return minkowski_white(sample_valid_pixels(image, self._max_pixels), 1.0)


class ShadesOfGrayEstimator(_StatisticalEstimator):
    """Minkowski-norm generalization of gray world."""

    name = "shades_of_gray"

    def __init__(self, confidence: float, max_pixels: int, power: float) -> None:
        super().__init__(confidence, max_pixels)
        self._power = power

    def _white(self, image: LinearImage) -> NDArray[np.float64]:
        return minkowski_white(sample_valid_pixels(image, self._max_pixels), self._power)


class WhitePatchEstimator(_StatisticalEstimator):
    """Assumes the brightest unsaturated surfaces are white."""

    name = "white_patch"

    def __init__(self, confidence: float, max_pixels: int, percentile: float) -> None:
        super().__init__(confidence, max_pixels)
        self._percentile = percentile

    def _white(self, image: LinearImage) -> NDArray[np.float64]:
        return percentile_white(sample_valid_pixels(image, self._max_pixels), self._percentile)


class GrayEdgeEstimator(_StatisticalEstimator):
    """Assumes the average edge difference is achromatic (van de Weijer 2007)."""

    name = "gray_edge"

    def __init__(self, confidence: float, max_pixels: int, power: float, sigma: float) -> None:
        super().__init__(confidence, max_pixels)
        self._power = power
        self._sigma = sigma

    def _white(self, image: LinearImage) -> NDArray[np.float64]:
        edges = gradient_magnitudes(image, self._sigma, self._max_pixels)
        return minkowski_white(edges, self._power)


class AsShotEstimator:
    """Uses the camera's own as-shot white balance from the RAW metadata."""

    name = "as_shot"

    def __init__(self, confidence: float) -> None:
        self._confidence = confidence

    def estimate(self, request: EstimationInput) -> EstimationOutcome:
        white = request.image.metadata.as_shot_white_rgb
        if white is None:
            raise EstimationUnavailableError("image has no as-shot white balance")
        return EstimationOutcome(
            IlluminantEstimate(
                white_rgb=normalize_white(white), method=self.name, confidence=self._confidence
            )
        )
