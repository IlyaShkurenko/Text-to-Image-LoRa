# Model Tester

Minimal CUDA/local tester for FLUX text-to-image models and LoRA adapters.

## Setup

On Vast.ai, start from a CUDA/PyTorch template when possible. Then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If the template does not already include PyTorch with CUDA, install PyTorch first using the official selector for your CUDA version:

https://pytorch.org/get-started/locally/

Set your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"
```

For Black Forest Labs models, your Hugging Face account must have accepted the model terms.
`flux2-klein-uncensored` also requires access to `ponpoke/flux2-klein-9b-uncensored-text-encoder`.

## Usage

Set the prompt in `model_tester/config.py`:

```python
PROMPT = "Astronaut riding a horse"
```

Run with the prompt from code:

```bash
python3 -m model_tester --device cuda
```

Pick a model from the registry:

```bash
python3 -m model_tester --device cuda --model flux-lustly
```

Base model without LoRA:

```bash
python3 -m model_tester --device cuda --model flux-dev
```

FLUX.2 Klein 9B with the ponpoke text encoder:

```bash
python3 -m model_tester --device cuda --model flux2-klein-uncensored
```

The `flux2-klein-uncensored` profile also loads the local FLUX.2 adapter stack from `models/loras/`.
SNOFS is applied as a LoKr merge, and Lenovo UltraReal is loaded as a standard LoRA:

```bash
python3 -m model_tester \
  --device cuda \
  --model flux2-klein-uncensored \
  --steps 20 \
  --guidance-scale 2.0
```

The FLUX.2 Klein profile defaults to `--steps 4`, `--guidance-scale 1.0`, and CUDA CPU offload. You can override them:

```bash
python3 -m model_tester --device cuda --model flux2-klein-uncensored --steps 8 --guidance-scale 1.0 --no-cpu-offload
```

For FLUX.2 Klein, Diffusers may print that guidance scale is ignored. That is expected for this step-wise
distilled model; tune prompt, seed, resolution, and step count instead.

Generated images and matching metadata JSON files are saved to `outputs/`.

By default, each run uses a random seed and writes the actual seed to the metadata JSON. To reproduce a result, pass the same seed explicitly:

```bash
python3 -m model_tester --device cuda --model flux-lustly --seed 42
```

For LoRA profiles, you can weaken or strengthen the adapter without editing code:

```bash
python3 -m model_tester --device cuda --model flux-lustly --lora-weight 0.6
```

You can also attach a LoRA by parameter instead of adding it to `model_tester/models.py`:

```bash
python3 -m model_tester \
  --device cuda \
  --model flux2-klein-uncensored \
  --lora-source REPLACE_WITH_REAL_HF_REPO \
  --lora-weight-name REPLACE_WITH_REAL_FILE.safetensors \
  --lora-adapter-name subject \
  --lora-weight 0.4
```

If you do not have a real FLUX.2-compatible LoRA yet, omit all `--lora-*` options and run only the base
`flux2-klein-uncensored` profile.

Use only LoRAs and prompts you have the legal right and consent to use.

## Vast.ai Quick Check

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If CUDA is available, run:

```bash
python3 -m model_tester --device cuda --model flux-lustly
```
