import sys
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image

COMMFOR_DIR = Path(__file__).parent / "Community-Forensics"
if str(COMMFOR_DIR) not in sys.path:
    sys.path.append(str(COMMFOR_DIR))

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

_MODEL_REPO = "OwensLab/commfor-model-384"
_PROCESSOR_REPO = "OwensLab/commfor-data-preprocessor"


def load_commfor_model():
    from models import ViTClassifier
    model = ViTClassifier.from_pretrained(_MODEL_REPO, device=str(device)).to(device)
    model.eval()
    print(f"[✓] Loaded Community Forensics model from {_MODEL_REPO}")
    return model


def load_commfor_processor():
    from dataprocessor_hf import CommForImageProcessor
    processor = CommForImageProcessor.from_pretrained(_PROCESSOR_REPO, size=384)
    print(f"[✓] Loaded Community Forensics processor from {_PROCESSOR_REPO}")
    return processor


@torch.inference_mode()
def predict_commfor(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    threshold: float = 0.5,
) -> Tuple[int, float]:
    processed = processor(image.convert("RGB"), mode="test")
    pixel_values = processed["pixel_values"].unsqueeze(0).to(device)
    logits = model(pixel_values)
    prob_fake = float(torch.sigmoid(logits).squeeze().item())
    label = 1 if prob_fake >= threshold else 0
    return label, prob_fake
