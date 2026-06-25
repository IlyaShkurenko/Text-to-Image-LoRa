import os
from pathlib import Path


HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN = os.environ.get(HF_TOKEN_ENV, "")

DEFAULT_MODEL_KEY = "flux-lustly"
PROMPT = "An amateur phone selfie of a woman sitting casually on a living room sofa. Natural look, imperfect skin texture with pores and subtle blemishes, unposed and authentic expression. The photo has a grainy quality with slight motion blur, captured with a harsh smartphone camera flash, creating realistic shadows on the wall behind her. She is fully exposed, showcasing detailed and anatomically accurate female genitalia, amateur photography style, low-key indoor lighting, captured on an older iPhone."
OUTPUT_DIR = Path("outputs")

DEVICE = "auto"
TORCH_DTYPE = "bfloat16"
GUIDANCE_SCALE = 4.0
HEIGHT = 768
WIDTH = 768
NUM_INFERENCE_STEPS = 20
SEED = 42
