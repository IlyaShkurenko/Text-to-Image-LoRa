import argparse
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
from .models import IMAGE_MODELS, ImageModel


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--seed", type=int, default=SEED, help="Use a fixed seed. Omit for a random seed.")
    return parser.parse_args()


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


def create_pipeline(model: ImageModel, device: str, dtype_name: str) -> tuple[object, str]:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
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
            f"Warning: {HF_TOKEN_ENV} is not set. FLUX.1-dev is gated, so loading may fail "
            "unless you already ran `huggingface-cli login` on this machine.",
            file=sys.stderr,
        )

    pipeline = AutoPipelineForText2Image.from_pretrained(
        model.base_model_id,
        torch_dtype=dtype,
        token=token,
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
            adapter_weights=[model.lora.adapter_weight],
        )

    return pipeline, resolved_device


def build_output_path(output_dir: str, model_key: str, prompt: str) -> str:
    safe_prompt = re.sub(r"[^a-zA-Z0-9]+", "-", prompt).strip("-").lower()[:50]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}-{model_key}-{safe_prompt or 'image'}.png"

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
) -> dict[str, object]:
    metadata = {
        "prompt": prompt,
        "output_path": output_path,
        "model_key": model.key,
        "base_model_id": model.base_model_id,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "seed": seed,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "torch_dtype": args.dtype,
    }

    if model.lora:
        metadata["lora"] = {
            "source": model.lora.source,
            "weight_name": model.lora.weight_name,
            "adapter_name": model.lora.adapter_name,
            "adapter_weight": model.lora.adapter_weight,
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

    pipeline, resolved_device = create_pipeline(model=model, device=args.device, dtype_name=args.dtype)
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipeline(
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=generator,
    )

    output_path = build_output_path(str(args.output_dir), model.key, prompt)
    image.images[0].save(output_path)
    save_metadata(output_path, build_metadata(model, prompt, args, output_path, resolved_device, seed))
    return output_path


def main() -> None:
    args = parse_args()
    model = IMAGE_MODELS[args.model]

    try:
        output_path = generate_image(model=model, prompt=args.prompt, args=args)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    print(f"Saved image: {output_path}")
