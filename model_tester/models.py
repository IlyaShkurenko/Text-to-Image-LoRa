from dataclasses import dataclass
from typing import Literal


PipelineKind = Literal["auto", "flux2-klein"]


@dataclass(frozen=True)
class LoraWeights:
    source: str
    adapter_name: str
    weight_name: str | None = None
    adapter_weight: float = 1.0


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
    ),
    "flux-dev": ImageModel(
        key="flux-dev",
        base_model_id="black-forest-labs/FLUX.1-dev",
    ),
    "flux2-klein-uncensored": ImageModel(
        key="flux2-klein-uncensored",
        base_model_id="black-forest-labs/FLUX.2-klein-9B",
        pipeline="flux2-klein",
        text_encoder=TextEncoderOverride(
            source="ponpoke/flux2-klein-9b-uncensored-text-encoder",
        ),
        default_guidance_scale=1.0,
        default_num_inference_steps=4,
        default_cpu_offload=True,
    ),
}
