"""Reading a ComfyUI model folder and saying which files can train.

ComfyUI and sd-scripts do not agree on what a "model" is. The folder layout usually
maps cleanly:

    models/checkpoints/       full checkpoints (SDXL, SD1.5)
    models/diffusion_models/  Flux transformers
    models/unet/              Flux transformers (older layout)
    models/text_encoders/     clip_l, t5xxl
    models/vae/               ae.safetensors, sdxl_vae.safetensors
    models/loras/             LoRAs, usable as initialisation only

What does not map is the *file*: an all-in-one Flux checkpoint has the transformer,
the text encoders and the VAE in one file, and sd-scripts wants the transformer
alone. A pre-quantised fp8 file is also not a training base, because `--fp8_base`
quantises from bf16 on the fly and a scaled fp8 file has already thrown that away.

Both are decidable from the safetensors header without torch, which is what this
module does: read the JSON header, look at the key prefixes and dtypes.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

MODEL_DIRS = {
    "checkpoints": "base",
    "unet": "base",
    "diffusion_models": "base",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "vae": "vae",
    "loras": "lora",
}

FLUX_PREFIXES = ("double_blocks.", "single_blocks.")
EMBEDDED_FLUX_PREFIX = "model.diffusion_model.double_blocks."
FP8_DTYPES = ("F8_E4M3", "F8_E5M2", "F8_E4M3FN", "F8_E4M3FNUZ")


@dataclass
class ModelFile:
    path: Path
    role: str
    folder: str
    size: int = 0
    kind: str = ""
    usable: bool = True
    note: str = ""


def find_models_dir(path: Path | str) -> Path:
    """Accept either a ComfyUI root containing ``models/`` or the folder itself."""
    root = Path(path).expanduser()
    candidate = root / "models"
    if candidate.is_dir():
        return candidate
    return root


def read_header(path: Path | str, *, max_bytes: int = 200_000_000) -> dict | None:
    """The JSON header of a safetensors file, or None when it is not one."""
    try:
        with open(path, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                return None
            (length,) = struct.unpack("<Q", prefix)
            if not 0 < length <= max_bytes:
                return None
            blob = handle.read(length)
            if len(blob) != length:
                return None
            header = json.loads(blob)
    except (OSError, json.JSONDecodeError, struct.error):
        return None

    if not isinstance(header, dict):
        return None
    header.pop("__metadata__", None)
    return header


def classify_header(header: dict) -> str:
    """What kind of weights a safetensors header describes."""
    keys = list(header)

    if any(key.startswith(prefix) for key in keys for prefix in FLUX_PREFIXES):
        return "flux_transformer"
    if any(key.startswith(EMBEDDED_FLUX_PREFIX) for key in keys):
        return "flux_embedded"
    if any(key.startswith("conditioner.embedders.") for key in keys):
        return "sdxl_checkpoint"
    if any("lora_down" in key or "lora_up" in key for key in keys):
        return "lora"
    if any(key.startswith("cond_stage_model.") for key in keys):
        return "sd15_checkpoint"
    if any(key.startswith("text_encoders.") for key in keys):
        return "embedded_text_encoder"
    if any(key.startswith("first_stage_model.") for key in keys):
        return "vae"
    if any(key.startswith("encoder.") for key in keys) and any(
        key.startswith("decoder.") for key in keys
    ):
        return "flux_vae"
    if any(key.startswith("model.diffusion_model.") for key in keys):
        return "unknown_checkpoint"
    return "unknown"


def _dtypes(header: dict, limit: int = 200) -> set[str]:
    found: set[str] = set()
    for index, meta in enumerate(header.values()):
        if index >= limit:
            break
        if isinstance(meta, dict) and isinstance(meta.get("dtype"), str):
            found.add(meta["dtype"])
    return found


def _role(folder: str, name: str) -> str:
    lower = name.lower()
    if folder in ("checkpoints", "unet", "diffusion_models"):
        return "base"
    if folder in ("text_encoders", "clip"):
        if "t5" in lower:
            return "t5xxl"
        if "clip_l" in lower or "clip-l" in lower:
            return "clip_l"
        return "text_encoder"
    if folder == "vae":
        return "vae"
    if folder == "loras":
        return "lora"
    return "unknown"


def validate(model: ModelFile) -> ModelFile:
    """Fill in kind and notes, and decide whether this file can train."""
    if model.role != "base":
        return model

    header = read_header(model.path)
    if header is None:
        model.usable = False
        model.note = "not a readable safetensors file"
        return model

    kind = classify_header(header)
    if kind == "flux_transformer":
        model.kind = "flux"
    elif kind == "sdxl_checkpoint":
        model.kind = "sdxl"
    elif kind == "sd15_checkpoint":
        model.kind = "sd15"
        model.usable = False
        model.note = "SD1.5 checkpoint; the presets here are SDXL and Flux"
    elif kind == "flux_embedded":
        model.usable = False
        model.note = (
            "all-in-one Flux file: the transformer is embedded with the text encoders "
            "and VAE. sd-scripts wants the transformer alone - use "
            "models/diffusion_models/flux1-dev.safetensors"
        )
    elif kind == "vae":
        model.usable = False
        model.note = "this is a VAE, not a checkpoint"
    else:
        model.usable = False
        model.note = f"unrecognised checkpoint layout ({kind})"

    if any(dtype.startswith(FP8_DTYPES) for dtype in _dtypes(header)):
        model.usable = False
        model.note = (
            (model.note + "; " if model.note else "")
            + "pre-quantised fp8 weights: --fp8_base quantises a bf16/bf16 copy on "
            "the fly, so keep the bf16 file as the training base"
        )

    return model


def scan(path: Path | str) -> list[ModelFile]:
    """Every model file under the known ComfyUI subfolders, validated."""
    models_dir = find_models_dir(path)
    found: list[ModelFile] = []

    for folder_name in MODEL_DIRS:
        folder = models_dir / folder_name
        if not folder.is_dir():
            continue
        for file in sorted(folder.iterdir()):
            if not file.is_file() or file.suffix.lower() not in (".safetensors", ".ckpt", ".pt"):
                continue
            model = ModelFile(
                path=file,
                role=_role(folder_name, file.name),
                folder=folder_name,
                size=file.stat().st_size,
            )
            if file.suffix.lower() == ".safetensors":
                validate(model)
            else:
                model.usable = False
                model.note = "legacy .ckpt/.pt format; convert to safetensors first"
            found.append(model)

    return found


def _prefer_t5(candidates: list[ModelFile]) -> list[ModelFile]:
    def key(model: ModelFile):
        name = model.path.name.lower()
        return ("fp8" in name, "fp16" not in name, model.path.name)

    return sorted(candidates, key=key)


def select(models: list[ModelFile], preset_name: str) -> dict[str, Path]:
    """Map a preset's required flags to the best usable files found."""
    chosen: dict[str, str | Path] = {}

    wanted_kind = "flux" if preset_name == "flux" else "sdxl"
    bases = [
        model for model in models if model.role == "base" and model.usable and model.kind == wanted_kind
    ]
    if bases:
        chosen["base"] = sorted(bases, key=lambda model: model.path.name)[0].path

    clip = [
        model
        for model in models
        if model.role == "clip_l" or (model.role == "text_encoder" and "clip_l" in model.path.name.lower())
    ]
    if clip:
        chosen["clip_l"] = clip[0].path

    t5 = [model for model in models if model.role == "t5xxl"]
    if t5:
        chosen["t5xxl"] = _prefer_t5(t5)[0].path

    vaes = [model for model in models if model.role == "vae"]
    if wanted_kind == "flux":
        ae = [model for model in vaes if model.path.stem.lower() == "ae"]
        if not ae:
            ae = vaes
        if ae:
            chosen["ae"] = ae[0].path
    else:
        sdxl_vae = [model for model in vaes if "sdxl" in model.path.name.lower()]
        if sdxl_vae:
            chosen["vae"] = sdxl_vae[0].path

    return {key: Path(value) for key, value in chosen.items()}
