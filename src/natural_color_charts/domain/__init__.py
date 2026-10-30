"""Immutable objects exchanged between pipeline stages."""

from natural_color_charts.domain.camera import CameraSensitivity
from natural_color_charts.domain.illuminant import (
    CorrectionTransform,
    IlluminantEstimate,
    normalize_white,
)
from natural_color_charts.domain.image import ImageMetadata, InputMode, LinearImage, PreviewImage
from natural_color_charts.domain.regions import (
    Detection,
    PatchMeasurement,
    ReferenceRegion,
    UserColorAnchor,
)
from natural_color_charts.domain.review import Critique, ReferenceEvidence, ReferenceVerdict
from natural_color_charts.domain.scene import (
    NEUTRAL_CLASSES,
    SCENE_CLASSES,
    AdaptationMode,
    BoxXyxy,
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
    "SCENE_CLASSES",
    "AdaptationMode",
    "BoxXyxy",
    "CameraSensitivity",
    "CorrectionTransform",
    "Critique",
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
    "ReferenceEvidence",
    "ReferenceRegion",
    "ReferenceVerdict",
    "RgbVector",
    "SceneContext",
    "SpectralArray",
    "UserColorAnchor",
    "UserIntent",
    "WavelengthGrid",
    "normalize_white",
    "resample_spectrum",
]
