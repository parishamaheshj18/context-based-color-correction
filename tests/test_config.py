import pytest
from pydantic import ValidationError

from natural_color_charts.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config.spectral.wavelength_step_nm == 10
    assert "spectral" in config.estimation.fallback_chain


def test_overrides_are_merged_recursively():
    config = load_config(overrides={"context": {"ollama": {"model": "other"}}})
    assert config.context.ollama.model == "other"
    assert config.context.ollama.host.startswith("http")


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        load_config(overrides={"render": {"unknown": 1}})
