"""Immutable objects exchanged between pipeline stages."""

from natural_color_charts.domain.camera import CameraSensitivity
from natural_color_charts.domain.illuminant import (
    CorrectionTransform,
    IlluminantEstimate,
    normalize_white,
)
from natural_color_charts.domain.image import ImageMetadata, InputMode, LinearImage, PreviewImage
from natural_color_charts.domain.regions import Detection, PatchMeasurement, ReferenceRegion
from natural_color_charts.domain.scene import (
    NEUTRAL_CLASSES,
    AdaptationMode,
    IlluminantHypothesis,
    ReferenceCandidate,
    ReferenceClass,
    SceneContext,
    UserIntent,
)
from natural_color_charts.domain.spectral import (
    Matrix3x3,
    RgbVector,
    SpectralArray,
    WavelengthGrid,
    resample_spectrum,
)

__all__ = [
    "NEUTRAL_CLASSES",
    "AdaptationMode",
    "CameraSensitivity",
    "CorrectionTransform",
    "Detection",
    "IlluminantEstimate",
    "IlluminantHypothesis",
    "ImageMetadata",
    "InputMode",
    "LinearImage",
    "Matrix3x3",
    "PatchMeasurement",
    "PreviewImage",
    "ReferenceCandidate",
    "ReferenceClass",
    "ReferenceRegion",
    "RgbVector",
    "SceneContext",
    "SpectralArray",
    "UserIntent",
    "WavelengthGrid",
    "normalize_white",
    "resample_spectrum",
]
