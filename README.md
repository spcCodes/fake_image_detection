# AI Fake Image Detection

A Streamlit web application that detects AI-generated images using an ensemble of two state-of-the-art deep learning models, classical forensic analysis, and optional GPT-4o Vision explanation.

## Overview

This tool combines three complementary detection approaches to identify AI-generated (AIGI) images:

| Detector | Source | Technique |
|---|---|---|
| **Effort** (×2 checkpoints) | ICML 2025 Oral | CLIP ViT-L/14 with SVD adaptation |
| **Community Forensics** | CVPR 2025 | Vision Transformer Small (ViT-S/16) |
| **ELA** | Classical forensics | JPEG recompression artifact analysis |

An optional **GPT-4o Vision** layer adds a human-readable forensic explanation on top of the numerical scores.

---

## Detection Methods

### 1. Effort (ICML 2025)
*"Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection"*

- **Backbone:** OpenAI CLIP ViT-L/14 (frozen)
- **Adaptation:** SVD decomposition on self-attention layers — only the low-rank residual subspace (rank 1023) is trainable, keeping the main singular component fixed
- **Two checkpoints:** trained on Stable Diffusion v1.4 and Chameleon generators
- **Output:** softmax probability → `prob_fake ∈ [0, 1]`

### 2. Community Forensics (CVPR 2025)
*"Community Forensics: Using Thousands of Generators to Train Fake Image Detectors"*

- **Backbone:** ViT-S/16 at 384×384 resolution
- **Training:** Thousands of diverse generators → strong cross-generator generalization
- **Augmentation:** RandomStateAugmentation (RSA) with stochastic JPEG, crop, flip, rotate, cutout ops
- **Output:** sigmoid score → `prob_fake ∈ [0, 1]`
- **Weights:** auto-downloaded from HuggingFace (`OwensLab/commfor-model-384`) on first run

### 3. Error Level Analysis (ELA)
Classical JPEG recompression forensic technique:

1. Re-save input as JPEG at quality 75, then subtract from original
2. Amplify differences ×15 to produce a visual error map
3. Compute Coefficient of Variation (CV) of error magnitudes:
   ```
   AI Score = 1 / (1 + CV)
   ```
- AI images → uniform synthesis → low CV → **high AI score**
- Real photos → content-dependent compression → high CV → **low AI score**
- Less reliable on PNG (no JPEG compression history)

### 4. GPT-4o Vision (Optional)
Sends the original image and ELA map to GPT-4o for structured forensic analysis:
- Verdict, confidence level, and rationale
- ELA map observations
- Visual artifact identification (symmetry, lighting, morphed objects, anomalous text)

---

## Ensemble (Combined Mode)

When **Combined** mode is selected, three scores (Effort-SDv1.4, Effort-Chameleon, Community Forensics) are aggregated:

| Strategy | Formula | Use Case |
|---|---|---|
| **Min / AND** | `min(s1, s2, s3)` | Minimize false positives (flag only if all agree) |
| **Average** | `mean(s1, s2, s3)` | Balanced vote |
| **Max / OR** | `max(s1, s2, s3)` | Minimize false negatives (flag if any detector is confident) |

ELA is always displayed alongside the ensemble result.

---

## Requirements

- Python ≥ 3.12
- PyTorch ≥ 2.0
- ~4–8 GB GPU/RAM for all models loaded simultaneously
- Supported input formats: JPEG, PNG, WebP, BMP, TIFF, HEIC

---

## Setup

### 1. Clone this repository

```bash
git clone <this-repo-url>
cd ai_fake_image_detection
```

### 2. Clone the external model repositories

`Community-Forensics` and `Effort-AIGI-Detection` are not bundled in this repo. Clone them manually into the project root:

```bash
git clone https://github.com/JeongsooP/Community-Forensics.git
git clone https://github.com/YZY-stack/Effort-AIGI-Detection.git
```

Your directory should then look like:

```
ai_fake_image_detection/
├── Community-Forensics/       ← cloned here
├── Effort-AIGI-Detection/     ← cloned here
├── app.py
└── ...
```

### 3. Install dependencies

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### 3. Download Effort model weights

Download the two checkpoints from the [Effort-AIGI-Detection repository](https://github.com/YZY-stack/Effort-AIGI-Detection) and place them in `weights/`:

```
weights/
├── effort_clip_L14_trainOn_sdv14.pth
└── effort_clip_L14_trainOn_chameleon.pth
```

Community Forensics weights are downloaded automatically from HuggingFace on first run.

### 4. (Optional) Set OpenAI API key

```bash
export OPENAI_API_KEY=sk-...
```

Or enter it directly in the app sidebar.

### 5. Run the app

```bash
streamlit run app.py
```

---

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. **Sidebar:** Select a detection mode and adjust the confidence threshold slider
3. **Upload** one or more images
4. View per-image results:
   - Verdict badge (Real / AI-Generated)
   - Score progress bar with confidence level (High / Medium / Low)
   - Individual model scores (Combined mode)
   - ELA map (expandable visualization)
   - GPT-4o forensic report (optional, requires API key)

**Confidence levels:**

| Score distance from threshold | Level |
|---|---|
| > 85% | High |
| 65–85% | Medium |
| < 65% | Low |

---

## Project Structure

```
ai_fake_image_detection/
├── app.py                    # Streamlit UI
├── inference.py              # Effort model loader + predict()
├── commfor_inference.py      # Community Forensics loader + predict_commfor()
├── ela_inference.py          # compute_ela() — ELA map + CV scoring
├── openai_analysis.py        # analyse_with_gpt4o() — GPT-4o integration
├── main.py                   # Entry point placeholder
├── pyproject.toml            # Dependencies
├── weights/                  # Effort model checkpoints (~1.2 GB each)
├── data/                     # Sample test images
├── Community-Forensics/      # CVPR 2025 model (cloned separately)
└── Effort-AIGI-Detection/    # ICML 2025 model (cloned separately)
    └── DeepfakeBench/training/
        ├── detectors/effort_detector.py   # EffortDetector, SVDResidualLinear
        └── config/detector/effort.yaml    # Training config
```

---

## References

- **Effort:** [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection) — ICML 2025 Oral
- **Community Forensics:** [JeongsooP/Community-Forensics](https://github.com/JeongsooP/Community-Forensics) — CVPR 2025
- **CLIP:** OpenAI ViT-L/14 via HuggingFace Transformers
- **Community Forensics weights:** `OwensLab/commfor-model-384` on HuggingFace
