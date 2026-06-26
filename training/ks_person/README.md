# ks_person LoRA training

Use this folder for a Flux2 Klein 9B identity LoRA trained with `ostris/ai-toolkit`.

Only use photos of an adult person who has consented to this training and the intended use.

## Dataset

Put images and matching captions in:

```text
training/ks_person/dataset/
  001.jpg
  001.txt
  002.jpg
  002.txt
```

Supported image formats are `jpg`, `jpeg`, and `png`. Keep the `.txt` filename identical to the image filename.

Every caption should contain the trigger:

```text
ks_person woman, ...
```

Describe the context that should not become part of identity: pose, crop, clothes, background, lighting, camera style.

## Caption Template

Send descriptions of your 12 photos in this format and I will turn them into final captions:

```text
001: close-up selfie, black top, bathroom mirror, phone visible, warm indoor light
002: half-body photo, sitting on sofa, white t-shirt, daylight from window
003: full-body photo, standing outdoors, black dress, evening street light
```

## Install ai-toolkit on Vast

Run outside this repo:

```bash
cd /workspace
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv venv
source venv/bin/activate
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Train

From the ai-toolkit folder:

```bash
cd /workspace/ai-toolkit
source venv/bin/activate
python run.py /workspace/Text-to-Image-LoRa/training/ks_person/config_flux2_klein_lora.yaml
```

The output LoRA checkpoints will be written under:

```text
/workspace/Text-to-Image-LoRa/training/ks_person/output/
```

After training, copy the best `.safetensors` file into `models/loras/` or point `model_tester` directly at the output file.
