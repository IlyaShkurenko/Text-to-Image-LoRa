import argparse
from collections import defaultdict
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
    parser.add_argument(
        "--lora-weight",
        type=float,
        action="append",
        default=None,
        help="Override LoRA weight. Repeat once per --lora-source for multiple adapters.",
    )
    parser.add_argument(
        "--lora-source",
        action="append",
        default=None,
        help="Optional Hugging Face repo or local path for LoRA weights. Repeat to stack adapters.",
    )
    parser.add_argument(
        "--lora-weight-name",
        action="append",
        default=None,
        help="Optional LoRA file name inside --lora-source. Repeat to match --lora-source.",
    )
    parser.add_argument(
        "--lora-adapter-name",
        action="append",
        default=None,
        help="Adapter name for --lora-source. Repeat to match --lora-source.",
    )
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


def profile_loras(model: ImageModel) -> list[LoraWeights]:
    loras = []
    if model.lora:
        loras.append(model.lora)
    loras.extend(model.loras)
    return loras


def get_indexed_str(values: list[str] | None, index: int) -> str | None:
    if not values:
        return None
    if index < len(values):
        return values[index]
    return None


def get_indexed_float(values: list[float] | None, index: int) -> float | None:
    if not values:
        return None
    if index < len(values):
        return values[index]
    return None


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
    for lora_source in args.lora_source or []:
        reject_placeholder_value("--lora-source", lora_source)
    for lora_weight_name in args.lora_weight_name or []:
        reject_placeholder_value("--lora-weight-name", lora_weight_name)

    if args.text_encoder_source:
        model = replace(model, text_encoder=TextEncoderOverride(source=args.text_encoder_source))

    if not args.lora_source and (args.lora_adapter_name or args.lora_weight_name):
        if not model.lora:
            raise RuntimeError("--lora-adapter-name and --lora-weight-name require --lora-source for this model.")

        model = replace(
            model,
            lora=replace(
                model.lora,
                adapter_name=get_indexed_str(args.lora_adapter_name, 0) or model.lora.adapter_name,
                weight_name=get_indexed_str(args.lora_weight_name, 0) or model.lora.weight_name,
            ),
        )

    return model


def resolve_loras(model: ImageModel, args: argparse.Namespace) -> list[LoraWeights]:
    if args.lora_source:
        loras = []
        for index, source in enumerate(args.lora_source):
            adapter_name = get_indexed_str(args.lora_adapter_name, index) or f"custom_{index + 1}"
            weight_name = get_indexed_str(args.lora_weight_name, index)
            adapter_weight = get_indexed_float(args.lora_weight, index)
            loras.append(
                LoraWeights(
                    source=source,
                    adapter_name=adapter_name,
                    weight_name=weight_name,
                    adapter_weight=adapter_weight if adapter_weight is not None else 1.0,
                )
            )
        return loras

    loras = profile_loras(model)
    if not loras:
        return []

    weights = args.lora_weight or []
    if not weights:
        return loras

    return [
        replace(lora, adapter_weight=weights[index] if index < len(weights) else lora.adapter_weight)
        for index, lora in enumerate(loras)
    ]


def resolve_lokr_target_modules(
    base_key: str,
    delta: object,
    modules: dict[str, object],
) -> list[tuple[object, object]]:
    def module(name: str) -> object | None:
        loaded = modules.get(name)
        if loaded is not None and hasattr(loaded, "weight"):
            return loaded
        return None

    direct = module(base_key)
    if direct is not None:
        return [(direct, delta)]

    parts = base_key.split(".")
    if len(parts) == 4 and parts[0] == "double_blocks":
        block_index = parts[1]
        stream = parts[2]
        layer = parts[3]
        prefix = f"transformer_blocks.{block_index}"

        if stream == "img_attn" and layer == "qkv":
            chunks = delta.chunk(3, dim=0)
            names = [f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"]
            return [(loaded, chunk) for name, chunk in zip(names, chunks) if (loaded := module(name)) is not None]
        if stream == "txt_attn" and layer == "qkv":
            chunks = delta.chunk(3, dim=0)
            names = [f"{prefix}.attn.add_q_proj", f"{prefix}.attn.add_k_proj", f"{prefix}.attn.add_v_proj"]
            return [(loaded, chunk) for name, chunk in zip(names, chunks) if (loaded := module(name)) is not None]

        mapped_names = {
            ("img_attn", "proj"): f"{prefix}.attn.to_out.0",
            ("txt_attn", "proj"): f"{prefix}.attn.to_add_out",
            ("img_mlp", "0"): f"{prefix}.ff.linear_in",
            ("img_mlp", "2"): f"{prefix}.ff.linear_out",
            ("txt_mlp", "0"): f"{prefix}.ff_context.linear_in",
            ("txt_mlp", "2"): f"{prefix}.ff_context.linear_out",
        }
        mapped = mapped_names.get((stream, layer))
        if mapped and (loaded := module(mapped)) is not None:
            return [(loaded, delta)]

    if len(parts) == 3 and parts[0] == "single_blocks":
        block_index = parts[1]
        layer = parts[2]
        mapped_names = {
            "linear1": f"single_transformer_blocks.{block_index}.attn.to_qkv_mlp_proj",
            "linear2": f"single_transformer_blocks.{block_index}.attn.to_out",
        }
        mapped = mapped_names.get(layer)
        if mapped and (loaded := module(mapped)) is not None:
            return [(loaded, delta)]

    return []


def apply_lokr_weights(pipeline: object, lora: LoraWeights) -> None:
    try:
        import torch
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("LoKr adapters require torch and safetensors.") from error

    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise RuntimeError("LoKr adapter loading expected pipeline.transformer, but it was not found.")

    modules = dict(transformer.named_modules())
    grouped_keys: dict[str, set[str]] = defaultdict(set)

    with safe_open(lora.source, framework="pt") as handle:
        for key in handle.keys():
            if not key.startswith("diffusion_model."):
                continue
            base_key, suffix = key.removeprefix("diffusion_model.").rsplit(".", 1)
            grouped_keys[base_key].add(suffix)

        applied = 0
        missing: list[str] = []
        unsupported: list[str] = []

        for base_key, suffixes in grouped_keys.items():
            if not {"lokr_w1", "lokr_w2"}.issubset(suffixes):
                unsupported.append(base_key)
                continue

            w1 = handle.get_tensor(f"diffusion_model.{base_key}.lokr_w1")
            w2 = handle.get_tensor(f"diffusion_model.{base_key}.lokr_w2")
            delta = torch.kron(w1.float(), w2.float()) * float(lora.adapter_weight)
            targets = resolve_lokr_target_modules(base_key, delta, modules)
            if not targets:
                missing.append(base_key)
                del delta, w1, w2
                continue

            for module, module_delta in targets:
                weight = module.weight
                module_delta = module_delta.reshape(weight.shape)
                with torch.no_grad():
                    weight.data.add_(module_delta.to(device=weight.device, dtype=weight.dtype))
                applied += 1

            del delta, w1, w2

    print(f"Applied LoKr adapter {lora.adapter_name}: {applied} modules from {lora.source}")
    if missing:
        print(
            f"Warning: skipped {len(missing)} LoKr modules with no matching transformer module. "
            f"First skipped: {missing[:5]}",
            file=sys.stderr,
        )
    if unsupported:
        print(
            f"Warning: skipped {len(unsupported)} unsupported LoKr groups. First skipped: {unsupported[:5]}",
            file=sys.stderr,
        )


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
    loras: list[LoraWeights],
    cpu_offload: bool,
) -> tuple[object, str]:
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        from transformers import AutoModelForCausalLM, AutoTokenizer
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
            pipeline_kwargs["text_encoder"] = AutoModelForCausalLM.from_pretrained(
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

    standard_loras = [lora for lora in loras if lora.adapter_kind == "lora"]
    lokr_loras = [lora for lora in loras if lora.adapter_kind == "lokr"]

    for lora in lokr_loras:
        apply_lokr_weights(pipeline, lora)

    for lora in standard_loras:
        lora_kwargs = {
            "adapter_name": lora.adapter_name,
            "token": token,
        }
        if lora.weight_name:
            lora_kwargs["weight_name"] = lora.weight_name

        pipeline.load_lora_weights(lora.source, **lora_kwargs)

    if standard_loras:
        pipeline.set_adapters(
            [lora.adapter_name for lora in standard_loras],
            adapter_weights=[lora.adapter_weight for lora in standard_loras],
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
    loras: list[LoraWeights],
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

    if loras:
        metadata["loras"] = [
            {
                "source": lora.source,
                "weight_name": lora.weight_name,
                "adapter_name": lora.adapter_name,
                "adapter_weight": lora.adapter_weight,
                "adapter_kind": lora.adapter_kind,
            }
            for lora in loras
        ]

    return metadata


def save_metadata(output_path: str, metadata: dict[str, object]) -> None:
    metadata_path = Path(output_path).with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_image(model: ImageModel, prompt: str, args: argparse.Namespace) -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Missing torch. Install CUDA PyTorch before running generation.") from error

    loras = resolve_loras(model, args)
    guidance_scale = resolve_guidance_scale(model, args.guidance_scale)
    num_inference_steps = resolve_num_inference_steps(model, args.steps)
    cpu_offload = resolve_cpu_offload(model, args.cpu_offload)
    pipeline, resolved_device = create_pipeline(
        model=model,
        device=args.device,
        dtype_name=args.dtype,
        loras=loras,
        cpu_offload=cpu_offload,
    )
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"Using seed: {seed}")
    print(f"Using steps: {num_inference_steps}")
    print(f"Using guidance scale: {guidance_scale}")
    if cpu_offload:
        print("Using CPU offload")
    for lora in loras:
        print(f"Using {lora.adapter_kind}: {lora.adapter_name} ({lora.source}) weight={lora.adapter_weight}")

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
            loras,
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
