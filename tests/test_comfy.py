import json
import struct

from loratrain.comfy import classify_header, read_header, scan, select


def _write(path, header, data=b"\x00" * 8):
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)
    return path


def _tensor(dtype="BF16"):
    return {"dtype": dtype, "shape": [1, 1], "data_offsets": [0, 2]}


FLUX_HEADER = {
    "double_blocks.0.img_attn.qkv.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
    "single_blocks.0.linear1.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
}
EMBEDDED_FLUX_HEADER = {
    "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [0, 2],
    },
    "text_encoders.t5xxl.transformer.shared.weight": {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [0, 2],
    },
}
SDXL_HEADER = {
    "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [0, 2],
    },
    "model.diffusion_model.input_blocks.0.0.weight": {
        "dtype": "BF16",
        "shape": [1],
        "data_offsets": [0, 2],
    },
}


def test_classify_flux_transformer():
    assert classify_header({"double_blocks.0.attn.weight": {}}) == "flux_transformer"


def test_classify_all_in_one_flux():
    assert classify_header(EMBEDDED_FLUX_HEADER) == "flux_embedded"


def test_classify_sdxl_checkpoint():
    assert classify_header(SDXL_HEADER) == "sdxl_checkpoint"


def test_classify_lora():
    assert classify_header({"lora_unet_x.lora_down.weight": {}}) == "lora"


def test_read_header_ignores_non_safetensors(tmp_path):
    junk = tmp_path / "junk.safetensors"
    junk.write_bytes(b"not a header")

    assert read_header(junk) is None


def _models_tree(tmp_path):
    diffusion = tmp_path / "models" / "diffusion_models"
    text = tmp_path / "models" / "text_encoders"
    vae = tmp_path / "models" / "vae"
    for folder in (diffusion, text, vae):
        folder.mkdir(parents=True)

    _write(diffusion / "flux1-dev.safetensors", FLUX_HEADER)
    _write(diffusion / "flux1-dev-fp8.safetensors", {"double_blocks.0.w": _tensor("F8_E4M3")})
    _write(text / "clip_l.safetensors", {"x.weight": _tensor()})
    _write(text / "t5xxl_fp16.safetensors", {"x.weight": _tensor()})
    _write(text / "t5xxl_fp8.safetensors", {"x.weight": _tensor()})
    _write(vae / "ae.safetensors", {"encoder.conv_in.weight": _tensor()})
    return tmp_path


def test_scan_finds_roles(tmp_path):
    root = _models_tree(tmp_path)

    found = scan(root)
    roles = {(model.role, model.path.name) for model in found}

    assert ("base", "flux1-dev.safetensors") in roles
    assert ("clip_l", "clip_l.safetensors") in roles
    assert ("t5xxl", "t5xxl_fp16.safetensors") in roles
    assert ("vae", "ae.safetensors") in roles


def test_all_in_one_flux_is_not_trainable(tmp_path):
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True)
    _write(root / "flux1-dev-all-in-one.safetensors", EMBEDDED_FLUX_HEADER)

    [model] = scan(tmp_path)

    assert model.usable is False
    assert "all-in-one" in model.note


def test_fp8_base_is_not_trainable(tmp_path):
    root = tmp_path / "models" / "diffusion_models"
    root.mkdir(parents=True)
    _write(root / "flux1-dev-fp8.safetensors", {"double_blocks.0.w": _tensor("F8_E4M3")})

    [model] = scan(tmp_path)

    assert model.usable is False
    assert "fp8" in model.note


def test_select_prefers_bf16_t5_and_the_flux_transformer(tmp_path):
    root = _models_tree(tmp_path)

    chosen = select(scan(root), "krea")

    assert chosen["base"].name == "flux1-dev.safetensors"
    assert chosen["t5xxl"].name == "t5xxl_fp16.safetensors"
    assert chosen["clip_l"].name == "clip_l.safetensors"
    assert chosen["ae"].name == "ae.safetensors"


def test_select_returns_nothing_for_a_preset_it_cannot_satisfy(tmp_path):
    root = tmp_path / "models" / "checkpoints"
    root.mkdir(parents=True)
    _write(root / "sd_xl_base_1.0.safetensors", SDXL_HEADER)

    chosen = select(scan(tmp_path), "flux")

    assert "base" not in chosen
