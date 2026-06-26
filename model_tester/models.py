from dataclasses import dataclass
from typing import Literal


PipelineKind = Literal["auto", "flux2-klein"]
AdapterKind = Literal["lora", "lokr", "aitoolkit_lora"]


@dataclass(frozen=True)
class LoraWeights:
    source: str
    adapter_name: str
    weight_name: str | None = None
    adapter_weight: float = 1.0
    adapter_kind: AdapterKind = "lora"
    prompt_prefix: str | None = None
    load_prefix: str | None = None
    use_load_prefix: bool = False
    state_dict_key_prefix: str | None = None
    strip_default_adapter_key: bool = False


@dataclass(frozen=True)
class TextEncoderOverride:
    source: str
    tokenizer_source: str | None = None


@dataclass(frozen=True)
class ImageModel:
    key: str
    base_model_id: str
    pipeline: PipelineKind = "auto"
    lora: LoraWeights | None = None
    loras: tuple[LoraWeights, ...] = ()
    text_encoder: TextEncoderOverride | None = None
    default_guidance_scale: float | None = None
    default_num_inference_steps: int | None = None
    default_cpu_offload: bool = False


IMAGE_MODELS: dict[str, ImageModel] = {
    "flux-lustly": ImageModel(
        key="flux-lustly",
        base_model_id="black-forest-labs/FLUX.1-dev",
        lora=LoraWeights(
            source="lustlyai/Flux_Lustly.ai_Uncensored_nsfw_v1",
            weight_name="flux_lustly-ai_v1.safetensors",
            adapter_name="v1",
            adapter_weight=1.0,
        ),
        loras=(
            LoraWeights(
                source="gorlamee/LenovoUltraReal",
                adapter_name="lenovo_ultrareal",
                adapter_weight=0.6,
            ),
        ),
    ),
    "flux-dev": ImageModel(
        key="flux-dev",
        base_model_id="black-forest-labs/FLUX.1-dev",
    ),
    "flux-dev-lenovo": ImageModel(
        key="flux-dev-lenovo",
        base_model_id="black-forest-labs/FLUX.1-dev",
        loras=(
            LoraWeights(
                source="gorlamee/LenovoUltraReal",
                adapter_name="lenovo_ultrareal",
                adapter_weight=0.8,
            ),
        ),
    ),
    "flux2-klein-uncensored": ImageModel(
        key="flux2-klein-uncensored",
        base_model_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline="flux2-klein",
        text_encoder=TextEncoderOverride(
            source="ponpoke/flux2-klein-9b-uncensored-text-encoder",
        ),
        loras=(
            LoraWeights(
                source="models/loras/klein_snofs_v1_4.safetensors",
                adapter_name="snofs",
                adapter_weight=0.8,
                adapter_kind="lokr",
            ),
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-uncensored-lenovo": ImageModel(
        key="flux2-klein-uncensored-lenovo",
        base_model_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline="flux2-klein",
        text_encoder=TextEncoderOverride(
            source="ponpoke/flux2-klein-9b-uncensored-text-encoder",
        ),
        loras=(
            LoraWeights(
                source="models/loras/klein_snofs_v1_4.safetensors",
                adapter_name="snofs",
                adapter_weight=0.8,
                adapter_kind="lokr",
            ),
            LoraWeights(
                source="Danrisi/Lenovo_FluxKlein9b_base",
                adapter_name="lenovo_klein",
                adapter_weight=0.6,
                prompt_prefix="l3n0v0.",
                use_load_prefix=True,
                state_dict_key_prefix="transformer.",
                strip_default_adapter_key=True,
            ),
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-uncensored-ks-lenovo": ImageModel(
        key="flux2-klein-uncensored-ks-lenovo",
        base_model_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline="flux2-klein",
        text_encoder=TextEncoderOverride(
            source="ponpoke/flux2-klein-9b-uncensored-text-encoder",
        ),
        loras=(
            LoraWeights(
                source="models/loras/klein_snofs_v1_4.safetensors",
                adapter_name="snofs",
                adapter_weight=0.8,
                adapter_kind="lokr",
            ),
            LoraWeights(
                source="models/loras/ks_person_flux2_klein_lora.safetensors",
                adapter_name="ks_person",
                adapter_weight=0.75,
                adapter_kind="aitoolkit_lora",
                prompt_prefix="ks_person woman,",
            ),
            LoraWeights(
                source="Danrisi/Lenovo_FluxKlein9b_base",
                adapter_name="lenovo_klein",
                adapter_weight=0.45,
                prompt_prefix="l3n0v0.",
                use_load_prefix=True,
                state_dict_key_prefix="transformer.",
                strip_default_adapter_key=True,
            ),
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-snofs-timdrnl": ImageModel(
        key="flux2-klein-snofs-timdrnl",
        base_model_id="timdrnl/FLUX.2-klein-9B-SNOFS",
        pipeline="flux2-klein",
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-snofs-timdrnl-lenovo": ImageModel(
        key="flux2-klein-snofs-timdrnl-lenovo",
        base_model_id="timdrnl/FLUX.2-klein-9B-SNOFS",
        pipeline="flux2-klein",
        loras=(
            LoraWeights(
                source="Danrisi/Lenovo_FluxKlein9b_base",
                adapter_name="lenovo_klein",
                adapter_weight=0.6,
                prompt_prefix="l3n0v0.",
                use_load_prefix=True,
                state_dict_key_prefix="transformer.",
                strip_default_adapter_key=True,
            ),
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-snofs-sintecs": ImageModel(
        key="flux2-klein-snofs-sintecs",
        base_model_id="sintecs/flux2-klein-9b-snofs-merged",
        pipeline="flux2-klein",
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
    "flux2-klein-snofs-sintecs-lenovo": ImageModel(
        key="flux2-klein-snofs-sintecs-lenovo",
        base_model_id="sintecs/flux2-klein-9b-snofs-merged",
        pipeline="flux2-klein",
        loras=(
            LoraWeights(
                source="Danrisi/Lenovo_FluxKlein9b_base",
                adapter_name="lenovo_klein",
                adapter_weight=0.6,
                prompt_prefix="l3n0v0.",
                use_load_prefix=True,
                state_dict_key_prefix="transformer.",
                strip_default_adapter_key=True,
            ),
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
}
