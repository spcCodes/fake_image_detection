import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image

from inference import load_model, predict, device
from commfor_inference import load_commfor_model, load_commfor_processor, predict_commfor
from ela_inference import compute_ela
from openai_analysis import analyse_with_gpt4o

load_dotenv()

EFFORT_CHECKPOINTS = {
    "Effort — GenImage / SDv1.4": {
        "path": "weights/effort_clip_L14_trainOn_sdv14.pth",
        "default_threshold": 0.5,
    },
    "Effort — Chameleon": {
        "path": "weights/effort_clip_L14_trainOn_chameleon.pth",
        "default_threshold": 0.5,
    },
}

DETECTION_MODES = [
    "Community Forensics",
    "Effort — GenImage / SDv1.4",
    "Effort — Chameleon",
    "Combined (all three)",
    "ELA (Error Level Analysis)",
]

COMBINE_MODES = {
    "Min / AND (reduce false positives)": "min",
    "Average (balanced)": "avg",
    "Max / OR (reduce false negatives)": "max",
}

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="centered",
)

st.title("AI-Generated Image Detector")
st.caption("Effort (ICML 2025 Oral) · CLIP ViT-L/14  +  Community Forensics (CVPR 2025) · ViT-S/16  +  ELA")
st.caption(f"Running on: `{device}`")

missing_repo = not (Path(__file__).parent / "Effort-AIGI-Detection").exists()
missing_commfor = not (Path(__file__).parent / "Community-Forensics").exists()
missing_weights = any(not Path(cfg["path"]).exists() for cfg in EFFORT_CHECKPOINTS.values())

if missing_repo or missing_weights or missing_commfor:
    st.error("**Setup required** — see instructions below.")
    with st.expander("Setup instructions", expanded=True):
        st.markdown("""
**1. Clone repos**
```bash
git clone https://github.com/YZY-stack/Effort-AIGI-Detection
git clone https://github.com/JeongsooP/Community-Forensics
```

**2. Download Effort checkpoints + CLIP model** (Google Drive link in Effort README)
```
weights/
├── effort_clip_L14_trainOn_sdv14.pth
└── effort_clip_L14_trainOn_chameleon.pth
```

**3. Community Forensics weights download automatically from HuggingFace on first run.**

**4. Install dependencies**
```bash
uv sync
```
""")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

detection_mode = st.sidebar.selectbox(
    "Detection model",
    DETECTION_MODES,
    index=0,
    help="Choose a single model or combine all three.",
)

st.sidebar.subheader("Threshold")

commfor_threshold = None
effort_thresholds = {}
combined_threshold = None
combine_mode = None
combine_label = None
ela_threshold = None

if detection_mode == "Community Forensics":
    commfor_threshold = st.sidebar.slider(
        "Community Forensics threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
    )

elif detection_mode in EFFORT_CHECKPOINTS:
    effort_thresholds[detection_mode] = st.sidebar.slider(
        f"{detection_mode} threshold",
        min_value=0.1, max_value=0.9,
        value=EFFORT_CHECKPOINTS[detection_mode]["default_threshold"],
        step=0.05,
    )

elif detection_mode == "ELA (Error Level Analysis)":
    ela_threshold = st.sidebar.slider(
        "ELA threshold",
        min_value=0.1, max_value=0.9, value=0.6, step=0.05,
        help="ELA score ≥ threshold → AI-Generated. Score is based on uniformity of the error level map.",
    )
    st.sidebar.caption(
        "**How it works:** Re-saves the image at JPEG quality 75, then measures "
        "how uniform the pixel-level differences are. A uniform map (low variance) "
        "suggests the whole image was synthesised at once → AI-generated.\n\n"
        "**Caveat:** PNG inputs have no prior JPEG history, so they compress "
        "uniformly regardless of origin — ELA is most reliable for JPEG images."
    )

else:  # Combined
    for name, cfg in EFFORT_CHECKPOINTS.items():
        effort_thresholds[name] = st.sidebar.slider(
            name,
            min_value=0.1, max_value=0.9,
            value=cfg["default_threshold"],
            step=0.05,
            key=f"thresh_{name}",
        )
    commfor_threshold = st.sidebar.slider(
        "Community Forensics",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        key="thresh_commfor",
    )
    st.sidebar.subheader("Combination method")
    combine_label = st.sidebar.selectbox("Method", list(COMBINE_MODES.keys()), index=0)
    combine_mode = COMBINE_MODES[combine_label]
    combined_threshold = st.sidebar.slider(
        "Combined threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Applied to the combined score to produce the final verdict.",
    )

# Always show ELA map toggle (except when ELA is the primary mode, it's always shown)
show_ela_map = detection_mode == "ELA (Error Level Analysis)" or st.sidebar.checkbox(
    "Show ELA forensic map", value=False,
    help="Display the Error Level Analysis map alongside the result for any detection mode.",
)

st.sidebar.divider()
st.sidebar.subheader("GPT-4o Vision Analysis")
use_gpt4o = st.sidebar.checkbox(
    "Enable GPT-4o analysis",
    value=False,
    help="Uses OpenAI GPT-4o Vision to analyse the original image and ELA map and explain its reasoning. Requires OPENAI_API_KEY.",
)
if use_gpt4o and not os.environ.get("OPENAI_API_KEY"):
    st.sidebar.warning("OPENAI_API_KEY not found. Set it in your .env file.")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Effort model weights…")
def get_effort_model(ckpt_path: str):
    return load_model(ckpt_path)


@st.cache_resource(show_spinner="Loading Community Forensics model (downloads on first run)…")
def get_commfor():
    model = load_commfor_model()
    processor = load_commfor_processor()
    return model, processor


effort_models = {name: get_effort_model(cfg["path"]) for name, cfg in EFFORT_CHECKPOINTS.items()}
commfor_model, commfor_processor = get_commfor()


# ── File upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload image(s)",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to analyse.")
    st.stop()


# ── Inference loop ────────────────────────────────────────────────────────────
for f in uploaded_files:
    img = Image.open(f)

    # Always compute ELA (needed for map display or as primary mode)
    ela_map, ela_score = compute_ela(img)

    if detection_mode == "Community Forensics":
        _, prob = predict_commfor(commfor_model, commfor_processor, img, threshold=commfor_threshold)
        score = prob
        threshold = commfor_threshold
        is_fake = score >= threshold

    elif detection_mode in EFFORT_CHECKPOINTS:
        _, prob = predict(effort_models[detection_mode], img, threshold=effort_thresholds[detection_mode])
        score = prob
        threshold = effort_thresholds[detection_mode]
        is_fake = score >= threshold

    elif detection_mode == "ELA (Error Level Analysis)":
        score = ela_score
        threshold = ela_threshold
        is_fake = score >= threshold

    else:  # Combined
        scores = {}
        for name, model in effort_models.items():
            _, prob = predict(model, img, threshold=effort_thresholds[name])
            scores[name] = prob
        _, commfor_prob = predict_commfor(commfor_model, commfor_processor, img, threshold=commfor_threshold)
        scores["Community Forensics"] = commfor_prob

        all_scores = list(scores.values())
        if combine_mode == "min":
            score = min(all_scores)
        elif combine_mode == "avg":
            score = sum(all_scores) / len(all_scores)
        else:
            score = max(all_scores)
        threshold = combined_threshold
        is_fake = score >= threshold

    verdict = "AI-Generated" if is_fake else "Real"
    badge_color = "red" if is_fake else "green"

    st.subheader(f"`{f.name}`")

    # ── Image + verdict row ───────────────────────────────────────────────────
    col_img, col_res = st.columns([1, 1])

    with col_img:
        st.image(img, use_container_width=True)

    with col_res:
        st.markdown(f"## :{badge_color}[{verdict}]")

        confidence = max(score, 1 - score)
        level = "High" if confidence > 0.85 else "Medium" if confidence > 0.65 else "Low"
        st.caption(f"Confidence: **{level}** ({confidence:.1%})")

        if detection_mode == "Combined (all three)":
            st.markdown("**Combined score**")
            st.progress(score)
            st.caption(f"{score:.1%}  ·  method: *{combine_label}*")
            st.markdown("**Individual scores**")
            for name, s in scores.items():
                t = commfor_threshold if name == "Community Forensics" else effort_thresholds[name]
                per_verdict = "AI" if s >= t else "Real"
                st.caption(f"{name}: `{s:.1%}` → {per_verdict}")
        elif detection_mode == "ELA (Error Level Analysis)":
            st.markdown("**ELA uniformity score**")
            st.progress(score)
            st.caption(
                f"{score:.1%}  ·  threshold: `{threshold:.2f}`  \n"
                "High score = uniform map = likely AI-generated."
            )
        else:
            st.markdown(f"**{detection_mode} score**")
            st.progress(score)
            st.caption(f"{score:.1%}  ·  threshold: `{threshold:.2f}`")

    # ── ELA forensic map ──────────────────────────────────────────────────────
    if show_ela_map:
        with st.expander("ELA forensic map", expanded=(detection_mode == "ELA (Error Level Analysis)")):
            ela_col1, ela_col2 = st.columns([1, 1])
            with ela_col1:
                st.image(ela_map, caption="ELA map (amplified ×15)", width='stretch')
            with ela_col2:
                st.markdown(
                    "**Reading the map**\n\n"
                    "- **Uniform / dark grey** — consistent error levels throughout; "
                    "typical of AI-generated images.\n"
                    "- **Bright white / yellow hotspots** — regions resaved more times "
                    "or copy-pasted; typical of edited real photos.\n"
                    "- **Natural variation** — expected in authentic unedited photos."
                )
                st.caption(f"ELA uniformity score: `{ela_score:.1%}`")

    # ── GPT-4o Vision Analysis ────────────────────────────────────────────────
    if use_gpt4o and os.environ.get("OPENAI_API_KEY"):
        with st.expander("GPT-4o Vision Analysis", expanded=True):
            if st.button("Run GPT-4o analysis", key=f"gpt4o_{f.name}"):
                with st.spinner("Sending to GPT-4o Vision…"):
                    try:
                        result = analyse_with_gpt4o(img, ela_map, ela_score)

                        gpt_verdict = result.get("verdict", "Unknown")
                        gpt_confidence = result.get("confidence", "Unknown")
                        gpt_color = "red" if gpt_verdict == "AI-Generated" else "green"

                        st.markdown(f"### :{gpt_color}[{gpt_verdict}]  ·  Confidence: {gpt_confidence}")

                        st.markdown("**Rationale**")
                        st.write(result.get("rationale", "—"))

                        col_ela, col_vis = st.columns(2)
                        with col_ela:
                            st.markdown("**ELA map observations**")
                            st.write(result.get("ela_observations", "—"))
                        with col_vis:
                            st.markdown("**Visual artifacts**")
                            st.write(result.get("visual_artifacts", "—"))

                        indicators = result.get("key_indicators", [])
                        if indicators:
                            st.markdown("**Key indicators**")
                            for item in indicators:
                                st.markdown(f"- {item}")

                    except Exception as e:
                        st.error(f"GPT-4o analysis failed: {e}")

    st.divider()
