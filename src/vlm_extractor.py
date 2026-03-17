

import re
import json
import io
import torch
from PIL import Image
from dataclasses import dataclass

from .text_extractor import ExtractionResult


# ── Prompts ────────────────────────────────────────────────────────────────────

IMAGE_EXTRACTION_PROMPT = """You are an expert information extractor for scientific and environmental documents.

Analyze this image carefully. It may be a map, chart, diagram, photograph, or technical figure.

Extract the following and respond ONLY with valid JSON (no markdown, no extra text):
{
  "image_type": "map | chart | diagram | photograph | table | other",
  "description": "one sentence describing what the image shows",
  "entities": [
    {"text": "<entity name>", "type": "<LOCATION|ORGANIZATION|SPECIES|MEASURE|METHOD|OBJECT>"}
  ],
  "relations": [
    {"subject": "...", "predicate": "...", "object": "..."}
  ],
  "visual_only_info": "information visible in the image NOT expressible in text (spatial layout, colors, scales, etc.)"
}

Be precise. If the image is unclear or contains no extractable information, return empty lists.
"""

JOINT_PROMPT_TEMPLATE = """You are analyzing a page from a scientific document.

TEXT FROM THE PAGE:
\"\"\"
{text}
\"\"\"

Look at the accompanying image and extract ALL entities and relations across both modalities.
Respond ONLY with valid JSON:
{{
  "entities": [
    {{"text": "...", "type": "LOCATION|ORGANIZATION|SPECIES|MEASURE|METHOD|OBJECT", "modality": "text|visual|both"}}
  ],
  "relations": [
    {{"subject": "...", "predicate": "...", "object": "...", "modality": "text|visual|both"}}
  ],
  "summary": "2–3 sentence summary combining text and visual information",
  "cross_modal_alignments": [
    {{"text_mention": "...", "visual_element": "...", "alignment_note": "..."}}
  ]
}}
"""


# ── Mock backend  ────────────────────────────────────

def _mock_extract(image: Image.Image, text: str) -> dict:
    """Returns a realistic mock response for demo purposes."""
    return {
        "entities": [
            {"text": "Rhine River", "type": "LOCATION",     "modality": "both"},
            {"text": "riparian habitat restoration", "type": "METHOD", "modality": "visual"},
            {"text": "2.3 km²",    "type": "MEASURE",       "modality": "visual"},
            {"text": "TETRA Project", "type": "ORGANIZATION","modality": "text"},
            {"text": "Salix alba",  "type": "SPECIES",       "modality": "visual"},
        ],
        "relations": [
            {"subject": "TETRA Project",    "predicate": "restoredAt",   "object": "Rhine River",    "modality": "both"},
            {"subject": "Salix alba",        "predicate": "foundIn",      "object": "riparian habitat", "modality": "visual"},
            {"subject": "riparian habitat",  "predicate": "covers",       "object": "2.3 km²",        "modality": "visual"},
        ],
        "summary": (
            "The image shows a detailed restoration map of the Upper Rhine floodplain, "
            "with annotated zones indicating phased intervention areas. "
            "Spatial information from the map complements the textual description of restoration methods."
        ),
        "cross_modal_alignments": [
            {
                "text_mention":   "restoration zones along the Rhine",
                "visual_element": "colored polygons on the map",
                "alignment_note": "Map visually delineates the zones described in the report text.",
            }
        ],
        "image_type": "map",
    }


# ── Model loaders ──────────────────────────────────────────────────────────────

_MODEL_CACHE = {}

def _load_blip2():
    if "blip2" in _MODEL_CACHE:
        return _MODEL_CACHE["blip2"]
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    print("[VLM] Loading BLIP-2 (this may take a few minutes)…")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _MODEL_CACHE["blip2"] = (processor, model)
    return processor, model


def _load_llava():
    if "llava" in _MODEL_CACHE:
        return _MODEL_CACHE["llava"]
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print("[VLM] Loading LLaVA-1.5-7B (this may take a few minutes)…")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    _MODEL_CACHE["llava"] = (processor, model)
    return processor, model


# ── Core extraction ────────────────────────────────────────────────────────────

def _run_blip2(image: Image.Image, prompt: str) -> str:
    processor, model = _load_blip2()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512)
    return processor.decode(output[0], skip_special_tokens=True)


def _run_llava(image: Image.Image, prompt: str) -> str:
    processor, model = _load_llava()
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(model.device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    full = processor.decode(output[0], skip_special_tokens=True)
    # Strip the prompt echo
    if "[/INST]" in full:
        full = full.split("[/INST]")[-1].strip()
    return full


def _parse_json_response(raw: str) -> dict:
    """Extract and parse a JSON object from a (possibly noisy) model response."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Try to find JSON block
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    # Return empty structure on failure
    return {
        "entities": [], "relations": [], "summary": raw[:300],
        "cross_modal_alignments": [], "image_type": "unknown"
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_multimodal(
    image: Image.Image,
    text: str,
    backend: str = "mock",
) -> ExtractionResult:
    """
    Run VLM-based multimodal extraction on a (image, text) pair.

    Args:
        image:   PIL image of the document page or embedded figure.
        text:    Raw text from the same page.
        backend: "mock" | "blip2" | "llava"

    Returns:
        ExtractionResult with modality="multimodal"
    """
    if backend == "mock":
        parsed = _mock_extract(image, text)
    else:
        prompt = JOINT_PROMPT_TEMPLATE.format(text=text[:2000])  # truncate for safety
        if backend == "blip2":
            raw = _run_blip2(image, prompt)
        elif backend == "llava":
            raw = _run_llava(image, prompt)
        else:
            raise ValueError(f"Unknown backend: {backend}. Choose from mock, blip2, llava.")
        parsed = _parse_json_response(raw)

    entities  = parsed.get("entities", [])
    relations = parsed.get("relations", [])
    summary   = parsed.get("summary", "")

    # Ensure all entities have a 'source' field
    for e in entities:
        e.setdefault("source", "visual")
    for r in relations:
        r.setdefault("source", "visual")

    return ExtractionResult(
        entities=entities,
        relations=relations,
        summary=summary,
        modality="multimodal",
    )
