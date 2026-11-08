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
    # "sd-scripts" writes one TOML config; "musubi" writes a dataset TOML and argv.
    engine: str = "sd-scripts"
    # musubi flags are not uniform: LTX-2 calls the DiT --ltx2_checkpoint.
    model_flags: dict = field(default_factory=dict, compare=False)

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
        ),        Preset(
            name="krea2",
            title="Krea 2 (musubi-tuner) - character LoRA",
            script="krea2_train_network.py",
            unit_kind="epochs",
            units=9,
            save_every=1,
            resolution=1024,
            min_bucket_reso=1024,
            max_bucket_reso=1536,
            network_module="networks.lora_krea2",
            network_dim=32,
            network_alpha=32,
            lr=1e-4,
            text_encoder_lr=None,
            lr_scheduler="cosine",
            lr_warmup_steps=63,
            max_grad_norm=1.0,
            default_repeats=1,
            images=(30, 60),
            requires=("dit", "vae", "text_encoder"),
            sampling=True,
            supports_vae=False,
            notes=(
                "musubi-tuner, not sd-scripts: convrot_int8 is the fast path on a "
                "3090 (fp8 dequantises every forward), blocks_to_swap 12 fits 24 GB, "
                "and the latents have to be cached before the first step"
            ),
            extra={
                "convrot_int8": True,
                "blocks_to_swap": 12,
                "timestep_sampling": "krea2_shift",
                "weighting_scheme": "none",
                "save_state": True,
            },
            engine="musubi",
        ),
        Preset(
            name="h3",
            title="MiniMax H3 (musubi-tuner) - character LoRA, teacher matching",
            script="minimax_h3_train_network.py",
            unit_kind="steps",
            units=500,
            save_every=250,
            resolution=1024,
            min_bucket_reso=1024,
            max_bucket_reso=1536,
            network_module="networks.lora_minimax_h3",
            network_dim=32,
            network_alpha=32,
            lr=3e-4,
            text_encoder_lr=None,
            lr_scheduler="cosine",
            lr_warmup_steps=50,
            max_grad_norm=1.0,
            default_repeats=1,
            images=(30, 60),
            requires=("dit",),
            sampling=False,
            supports_vae=False,
            notes=(
                "video model trained one frame at a time: both VAEs and the Qwen3-VL "
                "outputs must be cached first, the pretrained int8 DiT needs "
                "--block_swap_h2d_only, and the teacher-matching loss is what makes "
                "the identity hold instead of drifting"
            ),
            extra={
                "task": "t2va",
                "one_frame": True,
                "video_only": True,
                "blocks_to_swap": 48,
                "block_swap_h2d_only": True,
                "h3_teacher_matching": True,
                "h3_teacher_conditions": "subject_ref",
                "h3_teacher_condition_sigma_min": 0.15,
                "h3_teacher_loss_mag_weight": 0.5,
                "h3_teacher_loss_dc_weight": 0.3,
            },
            engine="musubi",
        ),
        Preset(
            name="ltx2",
            title="LTX-2 / 2.3 (musubi-tuner ltx-2 fork) - video LoRA",
            script="ltx2_train_network.py",
            unit_kind="steps",
            units=1000,
            save_every=250,
            resolution=1024,
            min_bucket_reso=768,
            max_bucket_reso=1280,
            network_module="networks.lora_ltx2",
            network_dim=32,
            network_alpha=16,
            lr=1e-4,
            text_encoder_lr=None,
            lr_scheduler="cosine",
            lr_warmup_steps=50,
            max_grad_norm=1.0,
            default_repeats=1,
            images=(20, 60),
            requires=("dit", "text_encoder"),
            sampling=False,
            supports_vae=False,
            notes=(
                "LTX-2 is not in upstream musubi-tuner: this needs the ltx-2 fork "
                "(AkaneTendo25/musubi-tuner, branch ltx-2), which ships ltx2_train_network.py "
                "and networks.lora_ltx2; nf4_base is the 3090 fit, the checkpoint is passed "
                "as --ltx2_checkpoint and Gemma as --gemma_root"
            ),
            extra={
                "ltx2_mode": "video",
                "blocks_to_swap": 32,
                "nf4_base": True,
            },
            model_flags={"dit": "ltx2_checkpoint", "text_encoder": "gemma_root"},
            engine="musubi",
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
