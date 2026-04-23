import io
import numpy as np
from PIL import Image


def compute_ela(image: Image.Image, quality: int = 75, amplify: int = 15) -> tuple[Image.Image, float]:
    """
    Returns (ela_map, ai_score).

    ELA score heuristic for AI detection:
      - AI images are synthesised uniformly, so re-compression produces consistent
        error levels across the whole frame → low coefficient of variation (CV).
      - Real JPEG photos have content-dependent error (texture > smooth regions) → higher CV.
      - Score = 1 / (1 + CV) on the RAW diff (before amplification/clipping).

    Note: PNG inputs (no JPEG history) compress uniformly on first save regardless
    of origin, so ELA is most discriminative for JPEG inputs.
    """
    img_rgb = image.convert("RGB")

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    orig = np.array(img_rgb, dtype=np.float32)
    comp = np.array(resaved, dtype=np.float32)

    diff_raw = np.abs(orig - comp)  # unclipped, for statistics

    # Visualisation only: amplify then clip
    diff_display = np.clip(diff_raw * amplify, 0, 255).astype(np.uint8)
    ela_map = Image.fromarray(diff_display)

    # Score: computed on raw diff to avoid clipping bias
    flat = diff_raw.flatten()
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    cv = std / (mean + 1e-6)

    # low CV → uniform error → likely AI-generated → score near 1
    score = 1.0 / (1.0 + cv)

    return ela_map, score
