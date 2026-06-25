from dataclasses import dataclass


@dataclass(frozen=True)
class LoraWeights:
    source: str
    adapter_name: str
    weight_name: str | None = None
    adapter_weight: float = 1.0


@dataclass(frozen=True)
class ImageModel:
    key: str
    base_model_id: str
    lora: LoraWeights | None = None


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
}
