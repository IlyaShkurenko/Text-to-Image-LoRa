import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import random
import re
import sys

from .config import (
    DEFAULT_MODEL_KEY,
    DEVICE,
    GUIDANCE_SCALE,
    HEIGHT,
    HF_TOKEN,
    HF_TOKEN_ENV,
    NUM_INFERENCE_STEPS,
    OUTPUT_DIR,
    PROMPT,
    SEED,
    TORCH_DTYPE,
    WIDTH,
)
from .models import IMAGE_MODELS, ImageModel, LoraWeights, TextEncoderOverride


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test local/CUDA diffusers text-to-image models.")
    parser.add_argument("--prompt", default=PROMPT, help="Override PROMPT from model_tester.config.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_KEY,
        choices=sorted(IMAGE_MODELS),
        help="Model key from model_tester.models.IMAGE_MODELS.",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for generated images.")
    parser.add_argument("--device", default=DEVICE, help='Device: "auto", "cuda", "mps", or "cpu".')
    parser.add_argument("--dtype", default=TORCH_DTYPE, help='Torch dtype: "bfloat16", "float16", or "float32".')
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps. Defaults to the selected model profile.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="Guidance scale. Defaults to the selected model profile.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Use a fixed seed. Omit for a random seed.")
    parser.add_argument("--lora-weight", type=float, default=None, help="Override the model profile LoRA weight.")
    parser.add_argument("--lora-source", default=None, help="Optional Hugging Face repo or local path for LoRA weights.")
    parser.add_argument("--lora-weight-name", default=None, help="Optional LoRA file name inside --lora-source.")
    parser.add_argument("--lora-adapter-name", default=None, help="Adapter name for --lora-source.")
    parser.add_argument(
        "--text-encoder-source",
        default=None,
        help="Override the selected profile's text encoder repo/local path.",
    )
    parser.add_argument(
        "--cpu-offload",
        dest="cpu_offload",
        action="store_true",
        default=None,
        help="Enable diffusers CPU offload for large models.",
    )
    parser.add_argument(
        "--no-cpu-offload",
        dest="cpu_offload",
        action="store_false",
        help="Disable model profile CPU offload.",
    )
    return parser.parse_args(argv)


def resolve_device(torch: object, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def resolve_dtype(torch: object, dtype_name: str) -> object:
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    try:
        return dtypes[dtype_name]
    except KeyError as error:
        allowed = ", ".join(sorted(dtypes))
        raise RuntimeError(f"Unknown dtype {dtype_name!r}. Use one of: {allowed}.") from error


def resolve_lora_weight(model: ImageModel, override_weight: float | None) -> float | None:
    if not model.lora:
        return None

    return model.lora.adapter_weight if override_weight is None else override_weight


def reject_placeholder_value(option_name: str, value: str | None) -> None:
    if not value:
        return

    placeholder_fragments = ("your-", "replace-me", "example/", "your_")
    if any(fragment in value for fragment in placeholder_fragments):
        raise RuntimeError(
            f"{option_name}={value!r} is a placeholder from the README. "
            "Replace it with a real Hugging Face repo/file name, or remove the LoRA options."
        )


def resolve_model(args: argparse.Namespace) -> ImageModel:
    model = IMAGE_MODELS[args.model]

    reject_placeholder_value("--text-encoder-source", args.text_encoder_source)
    reject_placeholder_value("--lora-source", args.lora_source)
    reject_placeholder_value("--lora-weight-name", args.lora_weight_name)

    if args.text_encoder_source:
        model = replace(model, text_encoder=TextEncoderOverride(source=args.text_encoder_source))

    if args.lora_source:
        model = replace(
            model,
            lora=LoraWeights(
                source=args.lora_source,
                adapter_name=args.lora_adapter_name or "custom",
                weight_name=args.lora_weight_name,
                adapter_weight=args.lora_weight if args.lora_weight is not None else 1.0,
            ),
        )
    elif args.lora_adapter_name or args.lora_weight_name:
        if not model.lora:
            raise RuntimeError("--lora-adapter-name and --lora-weight-name require --lora-source for this model.")

        model = replace(
            model,
            lora=replace(
                model.lora,
                adapter_name=args.lora_adapter_name or model.lora.adapter_name,
                weight_name=args.lora_weight_name or model.lora.weight_name,
            ),
        )

    return model


def resolve_guidance_scale(model: ImageModel, override_guidance_scale: float | None) -> float:
    if override_guidance_scale is not None:
        return override_guidance_scale

    if model.default_guidance_scale is not None:
        return model.default_guidance_scale

    return GUIDANCE_SCALE


def resolve_num_inference_steps(model: ImageModel, override_steps: int | None) -> int:
    if override_steps is not None:
        return override_steps

    if model.default_num_inference_steps is not None:
        return model.default_num_inference_steps

    return NUM_INFERENCE_STEPS


def resolve_cpu_offload(model: ImageModel, override_cpu_offload: bool | None) -> bool:
    if override_cpu_offload is not None:
        return override_cpu_offload

    return model.default_cpu_offload


def create_pipeline(
    model: ImageModel,
    device: str,
    dtype_name: str,
    lora_weight: float | None,
    cpu_offload: bool,
) -> tuple[object, str]:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Missing local generation dependencies. Install torch + diffusers first. "
            "On Vast.ai, use a CUDA PyTorch image and then run: "
            "python -m pip install -r requirements.txt"
        ) from error

    resolved_device = resolve_device(torch, device)
    dtype = resolve_dtype(torch, dtype_name)
    token = HF_TOKEN or None

    if not token and model.base_model_id.startswith("black-forest-labs/"):
        print(
            f"Warning: {HF_TOKEN_ENV} is not set. This Black Forest Labs model is gated, so loading may fail "
            "unless you already ran `huggingface-cli login` on this machine.",
            file=sys.stderr,
        )

    if model.pipeline == "flux2-klein":
        try:
            from diffusers import Flux2KleinPipeline
        except ImportError as error:
            raise RuntimeError(
                "The selected model needs diffusers.Flux2KleinPipeline. Install/upgrade Diffusers with: "
                "python -m pip install -U git+https://github.com/huggingface/diffusers.git transformers accelerate"
            ) from error

        pipeline_kwargs = {
            "torch_dtype": dtype,
            "token": token,
        }

        if model.text_encoder:
            tokenizer_source = model.text_encoder.tokenizer_source or model.text_encoder.source
            pipeline_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_source, token=token)
            pipeline_kwargs["text_encoder"] = AutoModel.from_pretrained(
                model.text_encoder.source,
                torch_dtype=dtype,
                token=token,
            )

        pipeline = Flux2KleinPipeline.from_pretrained(model.base_model_id, **pipeline_kwargs)
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            model.base_model_id,
            torch_dtype=dtype,
            token=token,
        )

    if cpu_offload and resolved_device == "cuda":
        if not hasattr(pipeline, "enable_model_cpu_offload"):
            raise RuntimeError("The selected pipeline does not support CPU offload.")
        pipeline.enable_model_cpu_offload()
    else:
        if cpu_offload:
            print(
                f"Warning: CPU offload is only enabled for CUDA; using pipeline.to({resolved_device!r}).",
                file=sys.stderr,
            )
        pipeline.to(resolved_device)

    if model.lora:
        lora_kwargs = {
            "adapter_name": model.lora.adapter_name,
            "token": token,
        }
        if model.lora.weight_name:
            lora_kwargs["weight_name"] = model.lora.weight_name

        pipeline.load_lora_weights(model.lora.source, **lora_kwargs)
        pipeline.set_adapters(
            [model.lora.adapter_name],
            adapter_weights=[lora_weight],
        )

    return pipeline, resolved_device


def build_output_path(output_dir: str, model_key: str, prompt: str, seed: int) -> str:
    safe_prompt = re.sub(r"[^a-zA-Z0-9]+", "-", prompt).strip("-").lower()[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{model_key}-seed-{seed}-{safe_prompt or 'image'}.png"

    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return str(output_path)


def build_metadata(
    model: ImageModel,
    prompt: str,
    args: argparse.Namespace,
    output_path: str,
    resolved_device: str,
    seed: int,
    lora_weight: float | None,
    guidance_scale: float,
    num_inference_steps: int,
    cpu_offload: bool,
) -> dict[str, object]:
    metadata = {
        "prompt": prompt,
        "output_path": output_path,
        "model_key": model.key,
        "base_model_id": model.base_model_id,
        "pipeline": model.pipeline,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "torch_dtype": args.dtype,
        "cpu_offload": cpu_offload,
    }

    if model.text_encoder:
        metadata["text_encoder"] = {
            "source": model.text_encoder.source,
            "tokenizer_source": model.text_encoder.tokenizer_source or model.text_encoder.source,
        }

    if model.lora:
        metadata["lora"] = {
            "source": model.lora.source,
            "weight_name": model.lora.weight_name,
            "adapter_name": model.lora.adapter_name,
            "adapter_weight": lora_weight,
        }

    return metadata


def save_metadata(output_path: str, metadata: dict[str, object]) -> None:
    metadata_path = Path(output_path).with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_image(model: ImageModel, prompt: str, args: argparse.Namespace) -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Missing torch. Install CUDA PyTorch before running generation.") from error

    lora_weight = resolve_lora_weight(model, args.lora_weight)
    guidance_scale = resolve_guidance_scale(model, args.guidance_scale)
    num_inference_steps = resolve_num_inference_steps(model, args.steps)
    cpu_offload = resolve_cpu_offload(model, args.cpu_offload)
    pipeline, resolved_device = create_pipeline(
        model=model,
        device=args.device,
        dtype_name=args.dtype,
        lora_weight=lora_weight,
        cpu_offload=cpu_offload,
    )
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"Using seed: {seed}")
    print(f"Using steps: {num_inference_steps}")
    print(f"Using guidance scale: {guidance_scale}")
    if cpu_offload:
        print("Using CPU offload")
    if lora_weight is not None:
        print(f"Using LoRA weight: {lora_weight}")

    image = pipeline(
        prompt=prompt,
        guidance_scale=guidance_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    output_path = build_output_path(str(args.output_dir), model.key, prompt, seed)
    image.images[0].save(output_path)
    save_metadata(
        output_path,
        build_metadata(
            model,
            prompt,
            args,
            output_path,
            resolved_device,
            seed,
            lora_weight,
            guidance_scale,
            num_inference_steps,
            cpu_offload,
        ),
    )
    return output_path


def main() -> None:
    args = parse_args()

    try:
        model = resolve_model(args)
        output_path = generate_image(model=model, prompt=args.prompt, args=args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Saved image: {output_path}")
