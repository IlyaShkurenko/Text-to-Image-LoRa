# LoRA Training Notes

Keep training separate from inference. This project should call trained LoRA files, not contain a custom trainer.

Recommended first path for Vast.ai:

1. Rent an NVIDIA GPU machine.
2. Install and run a mature FLUX LoRA trainer, such as `ostris/ai-toolkit`.
3. Train a LoRA adapter and export a `.safetensors` file.
4. Add a new profile in `model_tester/models.py` pointing to that file.

Expected dataset shape for common trainers:

```text
training/datasets/my_lora/
  image_001.jpg
  image_001.txt
  image_002.jpg
  image_002.txt
```

After training, add something like this to `IMAGE_MODELS`:

```python
"my-flux-lora": ImageModel(
    key="my-flux-lora",
    base_model_id="black-forest-labs/FLUX.1-dev",
    lora=LoraWeights(
        source="training/runs/my_lora/output",
        weight_name="my_lora.safetensors",
        adapter_name="my_lora",
        adapter_weight=1.0,
    ),
)
```
