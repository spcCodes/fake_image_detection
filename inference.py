"""
Inference wrapper for Effort AIGI Detection.

Directory layout expected (relative to this file):
    Effort-AIGI-Detection/
    └── DeepfakeBench/
        ├── models--openai--clip-vit-large-patch14/   ← CLIP ViT-L/14 weights
        └── training/
            ├── config/detector/effort.yaml
            └── detectors/effort_detector.py

    weights/
    ├── effort_genimage.pth
    └── effort_chameleon.pth

Download checkpoints + CLIP model from the Google Drive links in the repo README.
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from PIL import Image

REPO_TRAINING_DIR = Path(__file__).parent / "Effort-AIGI-Detection" / "DeepfakeBench" / "training"
EFFORT_CONFIG = REPO_TRAINING_DIR / "config" / "detector" / "effort.yaml"

if str(REPO_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_TRAINING_DIR))

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


@contextlib.contextmanager
def _chdir(path: Path):
    """Temporarily change working directory (needed for CLIP's relative model path)."""
    orig = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig)


def load_model(weights_path: str) -> torch.nn.Module:
    """Load Effort detector from a .pth checkpoint."""
    import yaml
    from detectors import DETECTOR

    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not EFFORT_CONFIG.exists():
        raise FileNotFoundError(
            f"Config not found at {EFFORT_CONFIG}. "
            "Clone the repo to Effort-AIGI-Detection/ first."
        )

    with open(EFFORT_CONFIG) as f:
        cfg = yaml.safe_load(f)

    # chdir so that effort_detector.py can resolve "../models--openai--clip-vit-large-patch14"
    with _chdir(REPO_TRAINING_DIR):
        model_cls = DETECTOR[cfg["model_name"]]
        model = model_cls(cfg).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[✓] Loaded checkpoint: {weights_path}")
    return model


@torch.inference_mode()
def predict(model: torch.nn.Module, image: Image.Image, threshold: float = 0.5) -> Tuple[int, float]:
    """
    Run inference on a PIL image.

    Returns:
        label    – 0 = Real, 1 = AI-Generated (Fake)
        prob_fake – float in [0, 1], probability the image is AI-generated
    """
    tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    data = {"image": tensor, "label": torch.tensor([0]).to(device)}
    preds = model(data, inference=True)
    prob_fake = float(preds["prob"].squeeze().item())
    label = 1 if prob_fake >= threshold else 0
    return label, prob_fake
