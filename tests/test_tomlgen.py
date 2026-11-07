from pathlib import Path

import pytest

from loratrain.presets import PRESETS
from loratrain.tomlgen import build_parts, render

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


MODELS = {
    "base": "/models/base.safetensors",
    "vae": None,
    "clip_l": "/models/clip_l.safetensors",
    "t5xxl": "/models/t5xxl.safetensors",
    "ae": "/models/ae.safetensors",
}


def _parts(preset_name="sdxl", trigger="mytok", repeats=8):
    preset = PRESETS[preset_name]
    return preset, build_parts(
        preset,
        dataset_dir=Path("/data/set"),
        out_dir=Path("/data/out"),
        name="test",
        models=MODELS,
        repeats=repeats,
        trigger=trigger,
        units=preset.units,
    )


@pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11+")
def test_rendered_config_is_valid_toml():
    preset, (top, general, dataset, subsets) = _parts()
    parsed = tomllib.loads(render(top, general, dataset, subsets))

    assert parsed["network_module"] == "networks.lora"
    assert parsed["max_train_epochs"] == preset.units
    assert parsed["datasets"][0]["resolution"] == 1024
    assert parsed["datasets"][0]["subsets"][0]["num_repeats"] == 8


@pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11+")
def test_flux_config_has_no_text_encoder_lr():
    preset, (top, general, dataset, subsets) = _parts("flux", repeats=1)
    parsed = tomllib.loads(render(top, general, dataset, subsets))

    assert parsed["network_module"] == "networks.lora_flux"
    assert "text_encoder_lr" not in parsed
    assert "max_train_steps" in parsed
    assert parsed["t5xxl_max_token_length"] == 512


def test_sdxl_samples_are_only_written_with_a_trigger():
    preset, (top, _general, _dataset, _subsets) = _parts(trigger=None)
    assert "sample_prompts" not in top
