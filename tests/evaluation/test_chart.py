import numpy as np

from natural_color_charts.colorimetry import angular_error_deg
from natural_color_charts.domain import ImageMetadata, InputMode, LinearImage
from natural_color_charts.evaluation.chart import ChartAnnotation, chart_white, sample_chart
from natural_color_charts.reporting.showcase import camera_label, hide_annotated_chart


def _chart_image(white):
    """A 24-patch grid whose gray row (patches 19-24) is lit by `white`."""
    rgb = np.zeros((400, 600, 3), np.float32)
    colors = np.random.default_rng(0).uniform(0.05, 0.5, (24, 3))
    colors[18:] = np.outer([0.9, 0.59, 0.36, 0.19, 0.09, 0.03], white)
    for index, color in enumerate(colors):
        row, column = divmod(index, 6)
        rgb[100 + row * 50 : 140 + row * 50, 100 + column * 60 : 150 + column * 60] = color
    image = LinearImage(rgb, np.zeros((400, 600), bool), ImageMetadata(), InputMode.RAW)
    corners = (
        (125 / 600, 120 / 400),
        (425 / 600, 120 / 400),
        (125 / 600, 270 / 400),
        (425 / 600, 270 / 400),
    )
    return image, ChartAnnotation(corners, (0.1, 0.2, 0.8, 0.8)), colors


def test_chart_sampling_recovers_patches_and_white():
    white = np.array([0.6, 1.0, 0.7])
    image, annotation, colors = _chart_image(white)
    patches = sample_chart(image, annotation)
    np.testing.assert_allclose(patches, colors, atol=1e-6)
    assert angular_error_deg(chart_white(patches), white) < 1e-6


def test_hidden_chart_is_black_and_excluded():
    image, annotation, _ = _chart_image(np.ones(3))
    hidden = hide_annotated_chart(image, annotation)
    assert hidden.excluded_mask[200, 300] and not hidden.valid_mask[200, 300]
    assert hidden.rgb[200, 300].max() == 0 and image.rgb[200, 300].max() > 0


def test_camera_label_drops_repeated_make():
    assert camera_label("Canon", "Canon EOS Rebel T8i") == "Canon EOS Rebel T8i"
    assert camera_label("Apple", "iPhone 16 Pro") == "Apple iPhone 16 Pro"
    assert camera_label(None, None) is None
