"""Open-vocabulary detection and SAM segmentation via Hugging Face transformers.

Requires the 'ml' extra (torch, transformers). Imports are deferred so the rest of the
package works without them, e.g. on machines without a GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from natural_color_charts.detection.base import DetectionRequest, PreviewImage, PreviewMask
from natural_color_charts.domain import Detection, ReferenceClass

logger = logging.getLogger(__name__)


def _require_ml() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Model-based detection needs the 'ml' extra: pip install '.[ml]'"
        ) from error


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection over union of two xyxy boxes."""
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def suppress_overlaps(
    detections: Sequence[Detection], iou_threshold: float, max_per_label: int
) -> list[Detection]:
    """Greedy per-label non-maximum suppression keeping at most max_per_label boxes."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: d.score, reverse=True):
        same_label = [k for k in kept if k.label == detection.label]
        if len(same_label) >= max_per_label:
            continue
        if all(box_iou(detection.box_xyxy, k.box_xyxy) < iou_threshold for k in same_label):
            kept.append(detection)
    return kept


class TransformersZeroShotDetector:
    """OWLv2 or Grounding DINO through the zero-shot-object-detection pipeline."""

    def __init__(
        self,
        model_id: str,
        device: str,
        prompts: Mapping[str, Sequence[str]],
        score_threshold: float,
        max_per_label: int,
        iou_threshold: float = 0.5,
    ) -> None:
        _require_ml()
        from transformers import pipeline

        self._pipeline: Any = pipeline("zero-shot-object-detection", model=model_id, device=device)
        self._prompts = prompts
        self._threshold = score_threshold
        self._max_per_label = max_per_label
        self._iou_threshold = iou_threshold

    def detect(self, request: DetectionRequest) -> list[Detection]:
        prompt_to_label = {
            prompt: label for label in request.labels for prompt in self._prompts.get(label, [])
        }
        if not prompt_to_label:
            return []
        height, width = request.preview.shape[:2]
        results = self._pipeline(
            Image.fromarray(request.preview),
            candidate_labels=list(prompt_to_label),
            threshold=self._threshold,
        )
        detections = [
            Detection(
                label=ReferenceClass(prompt_to_label[result["label"]]),
                box_xyxy=(
                    result["box"]["xmin"] / width,
                    result["box"]["ymin"] / height,
                    result["box"]["xmax"] / width,
                    result["box"]["ymax"] / height,
                ),
                score=float(result["score"]),
            )
            for result in results
            if result["label"] in prompt_to_label
        ]
        return suppress_overlaps(detections, self._iou_threshold, self._max_per_label)


class TransformersSamSegmenter:
    """Box-prompted Segment Anything; the model id selects SAM or a compatible checkpoint."""

    def __init__(self, model_id: str, device: str) -> None:
        _require_ml()
        from transformers import SamModel, SamProcessor

        self._device = device
        self._processor: Any = SamProcessor.from_pretrained(model_id)
        self._model: Any = SamModel.from_pretrained(model_id).to(device).eval()

    def segment(self, preview: PreviewImage, detections: Sequence[Detection]) -> list[PreviewMask]:
        if not detections:
            return []
        import torch

        height, width = preview.shape[:2]
        boxes = [
            [
                d.box_xyxy[0] * width,
                d.box_xyxy[1] * height,
                d.box_xyxy[2] * width,
                d.box_xyxy[3] * height,
            ]
            for d in detections
        ]
        inputs = self._processor(
            images=Image.fromarray(preview), input_boxes=[boxes], return_tensors="pt"
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, multimask_output=False)
        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]
        return [np.asarray(mask[0].numpy(), dtype=bool) for mask in masks]
