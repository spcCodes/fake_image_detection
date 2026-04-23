import base64
import io
import json
import os
from PIL import Image
from openai import OpenAI


def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def analyse_with_gpt4o(
    original_image: Image.Image,
    ela_map: Image.Image,
    ela_score: float,
) -> dict:
    """
    Sends the original image and its ELA map to GPT-4o Vision for forensic analysis.

    Returns a dict with keys:
        verdict, confidence, rationale, ela_observations, visual_artifacts, key_indicators
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    orig_b64 = _encode_image(original_image)
    ela_b64 = _encode_image(ela_map)

    system_prompt = (
        "You are an expert in digital image forensics specialising in detecting AI-generated images. "
        "You have deep knowledge of Error Level Analysis (ELA), GAN/diffusion model artifacts, "
        "lighting inconsistencies, texture anomalies, and other forensic indicators.\n\n"
        "ELA map interpretation guide:\n"
        "- Uniform / dark-grey map → consistent error levels across the whole frame → "
        "typical of AI-synthesised images (generated all at once).\n"
        "- Bright white / yellow hotspots → regions resaved multiple times or copy-pasted → "
        "common in edited real photos.\n"
        "- Natural spatial variation → expected in authentic, unedited photographs.\n\n"
        "Common AI-generation artifacts to look for in the original image:\n"
        "- Unnatural symmetry or over-smoothed skin/textures\n"
        "- Inconsistent lighting or impossible shadows\n"
        "- Malformed fingers, teeth, ears, or hair\n"
        "- Background objects that blend or morph unnaturally\n"
        "- Eye/iris anomalies (reflections, asymmetry)"
    )

    user_prompt = (
        f"Analyse the two images below for signs of AI generation.\n\n"
        f"Image 1 — Original image\n"
        f"Image 2 — Its Error Level Analysis (ELA) map (pixel differences after JPEG recompression, amplified ×15)\n\n"
        f"Pre-computed ELA uniformity score: {ela_score:.1%}  "
        f"(higher = more uniform error = more likely AI-generated; typical threshold ≈ 0.60)\n\n"
        "Respond ONLY with a JSON object using this exact schema:\n"
        "{\n"
        '  "verdict": "AI-Generated" or "Real",\n'
        '  "confidence": "High" or "Medium" or "Low",\n'
        '  "rationale": "2-3 sentence overall conclusion",\n'
        '  "ela_observations": "What the ELA map reveals and what it indicates",\n'
        '  "visual_artifacts": "Any visual artifacts or inconsistencies found in the original image",\n'
        '  "key_indicators": ["concise indicator 1", "concise indicator 2", ...]\n'
        "}"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{orig_b64}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{ela_b64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
    )

    return json.loads(response.choices[0].message.content)
