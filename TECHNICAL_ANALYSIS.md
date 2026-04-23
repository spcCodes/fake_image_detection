# AI Fake Image Detection — Technical Analysis

## 1. Project Overview

This is a **Streamlit-based web application** for detecting AI-generated images. It combines three complementary detection approaches — two state-of-the-art deep learning models from recent top-tier publications and one classical forensic technique — with an optional GPT-4o Vision explanation layer.

| Property | Value |
|---|---|
| Framework | PyTorch 2.0+, Streamlit |
| Python Version | ≥ 3.12 |
| GPU Support | CUDA, Apple Silicon (MPS), CPU fallback |
| Input Formats | JPEG, PNG, WebP, BMP, TIFF, HEIC |
| Model Memory | ~4–8 GB for all models loaded simultaneously |

---

## 2. Detection Methods

The app supports five detection modes, selectable from a sidebar dropdown.

### 2.1 Effort Detector (ICML 2025 Oral)

**Reference:** "Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection" (ICML 2025 Oral)

**Architecture:**
- Backbone: OpenAI CLIP ViT-L/14 (frozen)
- Head: `Linear(1024 → 2)` binary classification layer
- SVD Adaptation: `SVDResidualLinear` — applies rank-1023 SVD decomposition to CLIP self-attention layers; the model learns a residual update in the low-rank space while the backbone remains frozen
- Loss: CrossEntropyLoss with separate real/fake tracking; includes orthogonal loss and keep-SV loss regularization

**Input Preprocessing:**
- Resize to 224×224
- Normalize with CLIP statistics:
  - Mean: `[0.48145466, 0.4578275, 0.40821073]`
  - Std: `[0.26862954, 0.26130258, 0.27577711]`

**Checkpoint Variants (two provided):**
| Checkpoint | Trained On | File Size |
|---|---|---|
| `effort_clip_L14_trainOn_sdv14.pth` | Stable Diffusion v1.4 | ~1.2 GB |
| `effort_clip_L14_trainOn_chameleon.pth` | Chameleon generator | ~1.2 GB |

**Output:** Softmax probability → `prob_fake ∈ [0, 1]`; classified as AI-generated if `prob_fake > threshold`

**Config File:** `Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml`
- Optimizer: Adam, lr=0.0002, weight_decay=0.0005
- Augmentation: horizontal flip (p=0.5), rotation (±10°), blur (kernel 3–7), brightness/contrast (±0.1), JPEG quality (40–100)
- Batch size: 32, Epochs: 10, Metric: AUC
- Input: 224×224, 8 frames per video clip

---

### 2.2 Community Forensics (CVPR 2025)

**Reference:** "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors" (CVPR 2025)

**Architecture:**
- Backbone: Vision Transformer Small (ViT-S/16), patch size 16×16
- Input Resolution: 384×384
- Classification Head: `Linear(384 → 1)` with sigmoid activation
- Binary cross-entropy training

**Weights:** Auto-downloaded from HuggingFace on first run
| Asset | HuggingFace Repo |
|---|---|
| Model | `OwensLab/commfor-model-384` |
| Preprocessor | `OwensLab/commfor-data-preprocessor` |

**Output:** Sigmoid score → `prob_fake ∈ [0, 1]`; classified as AI-generated if `prob_fake > threshold`

**Implementation File:** `commfor_inference.py`
- `load_commfor_model()` — loads `ViTClassifier` from HuggingFace
- `load_commfor_processor()` — loads `CommForImageProcessor` (384×384)
- `predict_commfor(model, processor, image, threshold)` — returns `(label, prob_fake)`

---

### 2.3 ELA — Error Level Analysis (Classical Forensics)

**Technique:** JPEG recompression artifact analysis

**Algorithm (`ela_inference.py`):**
1. Save the input image as JPEG at quality 75
2. Reload and subtract from the original pixel-by-pixel to get an error map
3. Amplify differences by 15× for visualization
4. Score the image using the Coefficient of Variation (CV) of error magnitudes:

```
AI Score = 1 / (1 + CV)    where CV = std(errors) / mean(errors)
```

**Intuition:**
- AI-generated images have uniform pixel statistics → consistent JPEG error across regions → low CV → high AI score
- Real photographs have content-dependent compression error (edges vs. flat regions) → high CV → low AI score

**Output:** `(ela_map: PIL.Image, ai_score: float ∈ [0, 1])`

**Limitations:**
- Most reliable for JPEG images with authentic compression history
- PNG images (losslessly compressed, no JPEG history) compress uniformly regardless of origin, making ELA less discriminative
- Recommended threshold: ~0.60

---

## 3. GPT-4o Vision Integration (`openai_analysis.py`)

When enabled, the app sends both the original image and the ELA map to GPT-4o Vision for a structured forensic explanation.

**Workflow:**
1. Encode original image + ELA map as base64 JPEG (quality 90)
2. Provide system context: expert forensics role + ELA interpretation guide + known AI artifact signatures
3. Provide user prompt: structured JSON analysis request with pre-computed ELA score

**System prompt covers:**
- ELA map interpretation (bright regions = high error = likely edited/composited)
- Common AI artifacts: unnatural symmetry, impossible lighting, morphed background objects, eye/hair anomalies, garbled text

**Response Schema:**
```json
{
  "verdict": "AI-Generated | Real",
  "confidence": "High | Medium | Low",
  "rationale": "2–3 sentence conclusion",
  "ela_observations": "ELA map interpretation",
  "visual_artifacts": "Visual inconsistencies found",
  "key_indicators": ["indicator1", "indicator2", "..."]
}
```

**API Parameters:**
- Model: `gpt-4o`
- `response_format`: `json_object` (structured output)
- `max_tokens`: 1000
- Requires `OPENAI_API_KEY` environment variable

---

## 4. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                  Streamlit Web Interface (app.py)            │
│  Sidebar: mode, thresholds, combination strategy, GPT-4o     │
└────────────────────────┬─────────────────────────────────────┘
                         │
           ┌─────────────┼──────────────┐
           │             │              │
    ┌──────v──────┐ ┌────v──────────┐ ┌─v───────────────┐
    │   Effort    │ │  Community   │ │   ELA Analysis  │
    │  Detector   │ │  Forensics   │ │  (JPEG Recomp)  │
    │ CLIP-L/14   │ │  ViT-S/16    │ │  Classical      │
    │ (2 variants)│ │  384×384     │ │  Forensics      │
    └──────┬──────┘ └────┬──────────┘ └─┬───────────────┘
           │             │              │
           └─────────────┼──────────────┘
                         │
                 ┌───────v──────────┐
                 │  Score/Verdict   │
                 │  Aggregation     │
                 │  + Thresholding  │
                 └───────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ┌──────v──┐   ┌───────v──────┐  ┌───v─────────┐
   │ Verdict │   │   ELA Map    │  │  GPT-4o     │
   │ Display │   │ Visualization│  │  Analysis   │
   └─────────┘   └──────────────┘  └─────────────┘
```

---

## 5. Ensemble / Combined Mode

When the **Combined** detection mode is selected, all three deep learning scores are aggregated before thresholding.

**Individual scores:** `s1` (Effort-SDv1.4), `s2` (Effort-Chameleon), `s3` (Community Forensics)

| Strategy | Formula | Behaviour |
|---|---|---|
| **Min / AND** | `min(s1, s2, s3)` | Most conservative; flags only if all models agree (minimizes false positives) |
| **Average** | `mean(s1, s2, s3)` | Balanced probabilistic vote |
| **Max / OR** | `max(s1, s2, s3)` | Most aggressive; flags if any model is confident (minimizes false negatives) |

ELA can optionally be included in the display alongside the ensemble result.

---

## 6. Inference Pipeline

### Single Model Mode
```
Image Upload → Resize + Normalize → Model Forward Pass → prob_fake
             → Compare to threshold → Verdict (Real / AI-Generated)
```

### Combined Mode
```
Image Upload
    ├→ Effort-SDv1.4   → s1
    ├→ Effort-Chameleon → s2
    ├→ Community Forensics → s3
    └→ ELA              → ela_score (displayed separately)

Aggregation: combined_score = f(s1, s2, s3)  [min | avg | max]
Apply combined threshold → Final Verdict + Confidence Level
```

---

## 7. UI Configuration Options

| Setting | Type | Range / Options | Default | Effect |
|---|---|---|---|---|
| Detection Mode | Dropdown | 5 modes | Community Forensics | Selects pipeline |
| Threshold | Slider | 0.1–0.9 | Mode-dependent | Classification boundary |
| Show ELA Map | Checkbox | bool | False | Display forensic map |
| Combination Strategy | Dropdown | min / avg / max | min | Score aggregation function |
| GPT-4o Analysis | Checkbox | bool | False | Enable vision explanation |
| OpenAI API Key | Text input | sk-... | — | Required for GPT-4o |

### Confidence Levels (derived from `prob_fake`)
- **High:** score far from threshold in either direction
- **Medium:** score moderately separated from threshold
- **Low:** score close to threshold boundary

---

## 8. Key Source Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: upload, inference orchestration, display |
| `inference.py` | Effort model loader + `predict()` wrapper |
| `commfor_inference.py` | Community Forensics loader + `predict_commfor()` wrapper |
| `ela_inference.py` | `compute_ela()` — ELA map and CV-based score |
| `openai_analysis.py` | `analyse_with_gpt4o()` — GPT-4o structured report |
| `main.py` | Entry-point placeholder |
| `Community-Forensics/models.py` | `ViTClassifier` definition |
| `Community-Forensics/dataprocessor_hf.py` | `CommForImageProcessor` |
| `Community-Forensics/dataloader.py` | HuggingFace + folder-based dataset loaders |
| `Community-Forensics/custom_transforms.py` | Augmentation pipeline (RandAugment, StochasticJPEG, …) |
| `Community-Forensics/cf_utils.py` | Training / evaluation / checkpointing utilities |
| `Effort-AIGI-Detection/DeepfakeBench/training/detectors/effort_detector.py` | `EffortDetector`, `SVDResidualLinear` |
| `Effort-AIGI-Detection/DeepfakeBench/training/config/detector/effort.yaml` | Training hyperparameters and augmentation config |

---

## 9. Dependency Stack

**Core ML:**
- `torch ≥ 2.0`, `torchvision ≥ 0.15`
- `transformers ≥ 4.36` (CLIP via HuggingFace)
- `timm ≥ 0.9`, `efficientnet-pytorch ≥ 0.7.1`
- `huggingface-hub ≥ 1.11.0`

**Computer Vision:**
- `opencv-python ≥ 4.8`
- `Pillow ≥ 10.0`
- `scikit-image ≥ 0.22`

**Application:**
- `streamlit ≥ 1.32`
- `openai ≥ 1.30`
- `python-dotenv ≥ 1.0`

**Utilities:**
- `numpy ≥ 1.24`, `scikit-learn ≥ 1.3`, `pyyaml ≥ 6.0`
- `tqdm ≥ 4.66`, `imutils ≥ 0.5.4`, `wandb ≥ 0.26.0`
- `datasets ≥ 4.8.4`, `loralib ≥ 0.1.2`, `torchmetrics ≥ 1.9.0`

---

## 10. Setup and Deployment

### Prerequisites
```bash
# 1. Clone required model repositories
git clone https://github.com/YZY-stack/Effort-AIGI-Detection
git clone https://github.com/JeongsooP/Community-Forensics

# 2. Install dependencies (using uv)
uv sync
# or: pip install -r requirements.txt

# 3. Place Effort checkpoints in weights/
#    weights/effort_clip_L14_trainOn_sdv14.pth      (~1.2 GB)
#    weights/effort_clip_L14_trainOn_chameleon.pth  (~1.2 GB)

# 4. Community Forensics weights auto-download from HuggingFace on first run

# 5. (Optional) Set OpenAI API key for GPT-4o analysis
export OPENAI_API_KEY=sk-...
```

### Running the App
```bash
streamlit run app.py
```

---

## 11. Research Provenance

| Model | Publication | Venue |
|---|---|---|
| Effort | "Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection" | ICML 2025 (Oral) |
| Community Forensics | "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors" | CVPR 2025 |
| ELA | Standard digital forensics technique (no single publication) | Classical |
