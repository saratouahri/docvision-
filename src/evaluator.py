

from dataclasses import dataclass
from .text_extractor import ExtractionResult


@dataclass
class ComparisonReport:
    text_entity_count: int
    multimodal_entity_count: int
    visual_only_entity_count: int
    text_entity_types: dict      # {type: count}
    multimodal_entity_types: dict
    relation_text_count: int
    relation_multimodal_count: int
    visual_only_relation_count: int
    completeness_gain_pct: float  # % more entities from multimodal vs text-only
    overlap_pct: float            # % of text entities also found in multimodal
    new_entity_examples: list[str]


def compare_extractions(
    text_result: ExtractionResult,
    multimodal_result: ExtractionResult,
) -> ComparisonReport:
    """
    Produce a comparison report between text-only and multimodal extraction.
    """
    text_labels    = {e["text"].lower() for e in text_result.entities}
    mm_labels      = {e["text"].lower() for e in multimodal_result.entities}
    visual_only    = mm_labels - text_labels
    overlap        = text_labels & mm_labels

    completeness_gain = (
        ((len(mm_labels) - len(text_labels)) / max(len(text_labels), 1)) * 100
    )
    overlap_pct = (len(overlap) / max(len(text_labels), 1)) * 100

    # Type breakdowns
    def type_counts(result):
        counts = {}
        for e in result.entities:
            t = e.get("type", "UNKNOWN")
            counts[t] = counts.get(t, 0) + 1
        return counts

    text_rel_labels = {
        (r["subject"].lower(), r["predicate"], r["object"].lower())
        for r in text_result.relations
    }
    mm_rel_labels = {
        (r["subject"].lower(), r["predicate"], r["object"].lower())
        for r in multimodal_result.relations
    }
    visual_only_rels = mm_rel_labels - text_rel_labels

    new_examples = [
        e["text"] for e in multimodal_result.entities
        if e["text"].lower() in visual_only
    ][:5]

    return ComparisonReport(
        text_entity_count=len(text_result.entities),
        multimodal_entity_count=len(multimodal_result.entities),
        visual_only_entity_count=len(visual_only),
        text_entity_types=type_counts(text_result),
        multimodal_entity_types=type_counts(multimodal_result),
        relation_text_count=len(text_result.relations),
        relation_multimodal_count=len(multimodal_result.relations),
        visual_only_relation_count=len(visual_only_rels),
        completeness_gain_pct=round(completeness_gain, 1),
        overlap_pct=round(overlap_pct, 1),
        new_entity_examples=new_examples,
    )
