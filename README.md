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

The FLUX.2 Klein profile defaults to `--steps 4`, `--guidance-scale 1.0`, and CUDA CPU offload. You can override them:

```bash
python3 -m model_tester --device cuda --model flux2-klein-uncensored --steps 8 --guidance-scale 1.0 --no-cpu-offload
```

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
  --lora-source your-hf-user/your-flux2-lora \
  --lora-weight-name your_lora.safetensors \
  --lora-adapter-name subject \
  --lora-weight 0.4
```

Use only LoRAs and prompts you have the legal right and consent to use.

## Vast.ai Quick Check

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If CUDA is available, run:

```bash
python3 -m model_tester --device cuda --model flux-lustly
```
