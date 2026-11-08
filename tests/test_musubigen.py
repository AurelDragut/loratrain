from pathlib import Path

import pytest

from loratrain.musubigen import build_dataset, build_flags, flags_to_args, render_dataset
from loratrain.presets import PRESETS

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None


MODELS = {
    "dit": "/models/krea2.safetensors",
    "vae": "/models/vae.safetensors",
    "text_encoder": "/models/te.safetensors",
}


@pytest.mark.skipif(tomllib is None, reason="tomllib needs Python 3.11+")
def test_dataset_config_is_valid_toml_with_musubi_keys():
    preset = PRESETS["krea2"]
    general, datasets = build_dataset(
        preset, dataset_dir=Path("/data/set"), out_dir=Path("/data/out"), repeats=2
    )
    parsed = tomllib.loads(render_dataset(general, datasets))

    assert parsed["general"]["resolution"] == [1024, 1024]
    assert parsed["datasets"][0]["image_directory"] == "/data/set"
    assert parsed["datasets"][0]["cache_directory"] == "/data/out/cache"
    assert parsed["datasets"][0]["num_repeats"] == 2


def test_flags_carry_the_musubi_model_keys_first():
    preset = PRESETS["krea2"]

    args = flags_to_args(build_flags(preset, out_dir="/data/out", name="tok", models=MODELS, units=9))

    assert args[:6] == [
        "--dit",
        "/models/krea2.safetensors",
        "--vae",
        "/models/vae.safetensors",
        "--text_encoder",
        "/models/te.safetensors",
    ]
    assert "--convrot_int8" in args
    assert args[args.index("--max_train_epochs") + 1] == "9"


def test_true_switches_are_bare_and_false_ones_are_dropped():
    args = flags_to_args({"sdpa": True, "gradient_checkpointing": False, "blocks_to_swap": 12})

    assert args == ["--sdpa", "--blocks_to_swap", "12"]


def test_sampling_needs_a_trigger():
    preset = PRESETS["krea2"]

    without = flags_to_args(build_flags(preset, out_dir="/out", name="tok", models=MODELS))
    with_samples = flags_to_args(
        build_flags(
            preset,
            out_dir="/out",
            name="tok",
            models=MODELS,
            sample_prompts="/out/sample_prompts.txt",
        )
    )

    assert "--sample_prompts" not in without
    assert with_samples[with_samples.index("--sample_prompts") + 1] == "/out/sample_prompts.txt"


def test_krea2_is_a_musubi_preset():
    preset = PRESETS["krea2"]

    assert preset.engine == "musubi"
    assert preset.script == "krea2_train_network.py"
    assert preset.requires == ("dit", "vae", "text_encoder")
    assert PRESETS["sdxl"].engine == "sd-scripts"
