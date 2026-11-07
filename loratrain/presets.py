"""Per-model training presets.

Every number here exists because of a specific failure mode, and the ``notes``
field says which one. Presets are data, not behaviour: ``tomlgen`` turns one into
a config, so a preset is also the place to change what ``loratrain plan`` emits.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    title: str
    script: str
    unit_kind: str  # "epochs" or "steps"
    units: int
    save_every: int
    resolution: int
    min_bucket_reso: int
    max_bucket_reso: int
    network_module: str
    network_dim: int
    network_alpha: int
    lr: float
    text_encoder_lr: float | None
    lr_scheduler: str
    lr_warmup_steps: int
    max_grad_norm: float
    default_repeats: int
    images: tuple[int, int]
    requires: tuple[str, ...]
    sampling: bool
    supports_vae: bool
    notes: str
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def unit_flag(self) -> str:
        return "--epochs" if self.unit_kind == "epochs" else "--steps"


PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset(
            name="sdxl",
            title="SDXL base 1.0 - character or concept LoRA",
            script="sdxl_train_network.py",
            unit_kind="epochs",
            units=10,
            save_every=2,
            resolution=1024,
            min_bucket_reso=1024,
            max_bucket_reso=1536,
            network_module="networks.lora",
            network_dim=32,
            network_alpha=16,
            lr=1e-4,
            text_encoder_lr=5e-5,
            lr_scheduler="cosine",
            lr_warmup_steps=100,
            max_grad_norm=1.0,
            default_repeats=8,
            images=(40, 80),
            requires=("base",),
            sampling=True,
            supports_vae=True,
            notes=(
                "anything under 1024 on the short side loses eyes; above ~2e-4 "
                "faces burn; fp16 text encoders give NaN latents, so bf16 always"
            ),
            extra={
                "no_half_vae": True,
                "sdpa": True,
                "cache_latents": True,
                "cache_latents_to_disk": True,
                "min_snr_gamma": 5,
                "noise_offset": 0.0,
            },
        ),
        Preset(
            name="flux",
            title="FLUX.1 [dev] - character or concept LoRA",
            script="flux_train_network.py",
            unit_kind="steps",
            units=2000,
            save_every=250,
            resolution=1024,
            min_bucket_reso=1024,
            max_bucket_reso=1536,
            network_module="networks.lora_flux",
            network_dim=32,
            network_alpha=16,
            lr=1e-4,
            text_encoder_lr=None,
            lr_scheduler="cosine",
            lr_warmup_steps=50,
            max_grad_norm=1.0,
            default_repeats=1,
            images=(40, 120),
            requires=("base", "clip_l", "t5xxl", "ae"),
            sampling=False,
            supports_vae=False,
            notes=(
                "anatomy that occupies 30-80px in a 1024 frame cannot be learned; "
                "30-40% of the dataset should be crops where it is 250-500px"
            ),
            extra={
                "model_prediction_type": "raw",
                "guidance_scale": 1.0,
                "apply_t5_attn_mask": True,
                "timestep_sampling": "shift",
                "discrete_flow_shift": 3.0,
                "t5xxl_max_token_length": 512,
                "fp8_base": True,
                "split_mode": True,
                "cache_text_encoder_outputs": True,
                "cache_text_encoder_outputs_to_disk": True,
            },
        ),
        Preset(
            name="krea",
            title="FLUX.1 Krea [dev] - character or concept LoRA",
            script="flux_train_network.py",
            unit_kind="steps",
            units=3000,
            save_every=250,
            resolution=1024,
            min_bucket_reso=1024,
            max_bucket_reso=1536,
            network_module="networks.lora_flux",
            network_dim=32,
            network_alpha=16,
            lr=5e-5,
            text_encoder_lr=None,
            lr_scheduler="cosine",
            lr_warmup_steps=50,
            max_grad_norm=1.0,
            default_repeats=1,
            images=(30, 60),
            requires=("base", "clip_l", "t5xxl", "ae"),
            sampling=False,
            supports_vae=False,
            notes=(
                "train against the Krea transformer, not dev: a dev LoRA applied to "
                "Krea looks like the face never trained, and Krea burns earlier so "
                "the LR is half of flux's"
            ),
            extra={
                "model_prediction_type": "raw",
                "guidance_scale": 1.0,
                "apply_t5_attn_mask": True,
                "timestep_sampling": "flux_shift",
                "discrete_flow_shift": 3.0,
                "t5xxl_max_token_length": 512,
                "fp8_base": True,
                "split_mode": True,
                "cache_text_encoder_outputs": True,
                "cache_text_encoder_outputs_to_disk": True,
            },
        ),
    )
}


# Keys that mean the same thing under a different name in some sd-scripts
# versions. The first alias the installed version supports wins.
ALIASES: dict[str, tuple[str, ...]] = {
    "t5xxl_max_token_length": ("t5_max_token_length",),
}


def presets_text() -> str:
    lines: list[str] = []
    for preset in PRESETS.values():
        lines.append(f"{preset.name:<6} {preset.title}")
        lines.append(
            f"       {preset.script}, {preset.unit_kind} {preset.units}, "
            f"lr {preset.lr:g}, dim {preset.network_dim}/{preset.network_alpha}"
        )
        lines.append(
            f"       expects {preset.images[0]}-{preset.images[1]} images; "
            f"needs {', '.join(preset.requires)}"
        )
        lines.append(f"       {preset.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
