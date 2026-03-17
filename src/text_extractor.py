
import re
import json
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    entities: list[dict]          # [{"text": ..., "type": ...}]
    relations: list[dict]         # [{"subject": ..., "predicate": ..., "object": ...}]
    summary: str
    modality: str = "text-only"


# ── Simple rule-based NER  ─────────────────────────

ENTITY_PATTERNS = {
    "LOCATION":     r"\b([A-Z][a-z]+ (?:River|Lake|Forest|Valley|Mountain|Basin|Creek|Wetland|Delta|Estuary))\b",
    "ORGANIZATION": r"\b((?:[A-Z][a-z]* ){1,3}(?:Agency|Ministry|Institute|University|Laboratory|Project|Program))\b",
    "DATE":         r"\b(\d{4}(?:[-/]\d{2}[-/]\d{2})?)\b",
    "MEASURE":      r"\b(\d+(?:\.\d+)?\s*(?:km²?|ha|m²?|mg/L|μg/L|ppm|%|°C|tons?|species))\b",
    "METHOD":       r"\b((?:habitat|wetland|riparian|stream|river|floodplain)\s+(?:restoration|monitoring|assessment|management|rehabilitation))\b",
    "SPECIES": r"\b([A-Z][a-z]+ [a-z]+)\b(?=\s+(?:population|habitat|species|var\.|subsp\.|strain|spp\.))",
}

RELATION_PATTERNS = [
    (r"(?P<subj>[A-Z][a-z ]+)\s+(?:was|were)\s+restored\s+(?:in|at|near)\s+(?P<obj>[A-Z][a-z ]+)", "restoredAt"),
    (r"(?P<subj>[A-Z][a-z ]+)\s+(?:is|are)\s+located\s+(?:in|at|near)\s+(?P<obj>[A-Z][a-z ]+)",   "locatedIn"),
    (r"(?P<subj>[A-Z][a-z ]+)\s+(?:implemented|conducted|carried out)\s+(?P<obj>[a-z ]+(?:restoration|monitoring|assessment))", "implemented"),
    (r"(?P<subj>[A-Z][a-z ]+)\s+(?:measured|recorded|observed)\s+(?:at|as)\s+(?P<obj>\d[^.]+)", "measuredAs"),
]


def extract_text_only(text: str) -> ExtractionResult:
    """
    Rule-based extraction from raw text.
    Returns entities, relations, and a basic summary.
    """
    entities = []
    seen = set()

    for etype, pattern in ENTITY_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE if etype in ("METHOD",) else 0):
            val = match.group(1).strip()
            key = (val.lower(), etype)
            if key not in seen:
                seen.add(key)
                entities.append({"text": val, "type": etype, "source": "text"})

    relations = []
    for pattern, pred in RELATION_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            subj = match.group("subj").strip()
            obj  = match.group("obj").strip()
            if subj and obj:
                relations.append({
                    "subject":   subj,
                    "predicate": pred,
                    "object":    obj,
                    "source":    "text",
                })

    # Simple extractive summary: first 3 sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    summary = " ".join(sentences[:3]) if sentences else text[:300]

    return ExtractionResult(entities=entities, relations=relations, summary=summary, modality="text-only")


def merge_results(text_result: ExtractionResult, visual_result: ExtractionResult) -> ExtractionResult:
    """
    Merge text-only and multimodal results, deduplicating entities and relations.
    """
    seen_entities = set()
    merged_entities = []
    for e in text_result.entities + visual_result.entities:
        key = (e["text"].lower(), e["type"])
        if key not in seen_entities:
            seen_entities.add(key)
            merged_entities.append(e)

    seen_relations = set()
    merged_relations = []
    for r in text_result.relations + visual_result.relations:
        key = (r["subject"].lower(), r["predicate"], r["object"].lower())
        if key not in seen_relations:
            seen_relations.add(key)
            merged_relations.append(r)

    combined_summary = f"[TEXT] {text_result.summary}\n\n[VISUAL] {visual_result.summary}"

    return ExtractionResult(
        entities=merged_entities,
        relations=merged_relations,
        summary=combined_summary,
        modality="multimodal",
    )
