# DocVision 🔬
**Structured Information Extraction from Scientific PDFs using Vision-Language Models**
[![Open in HuggingFace Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Spaces-blue)](https://huggingface.co/spaces/sarathr19/docvision)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/sarathr19/docvision)

DocVision is a research pipeline that extracts entities and relations from heterogeneous scientific documents (reports, maps, technical drawings, photographs) by **jointly processing text and images** through a Vision-Language Model (VLM). It compares text-only vs. multimodal extraction and produces a queryable RDF knowledge graph.

---

## Research Motivation

Scientific and technical documents — environmental reports, restoration plans, engineering drawings — contain critical information split across **text** and **visual elements** (maps, diagrams, photographs, charts). Text-only NLP pipelines miss spatial layouts, legend annotations, and visual measurements entirely.

This project demonstrates that **multimodal extraction with VLMs** achieves:
- Higher entity completeness (especially for spatial/visual entities)
- Better cross-modal alignment of information
- Traceable, structured RDF output suitable for knowledge-based querying

---

## Architecture

```
PDF Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  pdf_parser.py  (PyMuPDF)                            │
│  ├── Raw text per page                               │
│  ├── Embedded images (figures, maps)                 │
│  └── Full-page render (high-DPI PIL image)           │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌────────────────┐   ┌──────────────────────┐
│ text_extractor │   │  vlm_extractor.py    │
│ (rule-based /  │   │  (BLIP-2 / LLaVA)   │
│  LLM baseline) │   │  image + text → JSON │
└───────┬────────┘   └──────────┬───────────┘
        └──────────┬────────────┘
                   ▼
          merge_results()
                   │
                   ▼
┌──────────────────────────────┐
│  kg_builder.py  (RDFLib)    │
│  entities + relations → RDF  │
│  SPARQL query interface      │
└──────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────┐
│  evaluator.py                │
│  text vs. multimodal metrics │
│  completeness, overlap, gain │
└──────────────────────────────┘
                   │
                   ▼
        Streamlit Dashboard
```

---

## Installation

```bash
# 1. Clone or download the project
cd docvision

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### GPU (optional but recommended for real VLMs)
```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Running the App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## VLM Backends

| Backend | Model | VRAM | Notes |
|---------|-------|------|-------|
| `mock`  | Built-in deterministic mock | 0 GB | Instant, great for demos and development |
| `blip2` | `Salesforce/blip2-opt-2.7b` | ~6 GB | Good balance of quality and speed |
| `llava` | `llava-hf/llava-1.5-7b-hf` | ~14 GB | Best extraction quality |

Start with `mock` to explore the UI, then switch to `blip2` or `llava` for real extraction.

---

## Project Structure

```
docvision/
├── app.py                  ← Streamlit dashboard (entry point)
├── requirements.txt        ← Python dependencies
├── src/
│   ├── __init__.py
│   ├── pdf_parser.py       ← PDF → pages (text + images)
│   ├── text_extractor.py   ← Baseline text-only extraction
│   ├── vlm_extractor.py    ← VLM multimodal extraction
│   ├── kg_builder.py       ← RDF knowledge graph construction
│   └── evaluator.py        ← Comparison metrics
└── outputs/                ← Saved .ttl / .json / .csv results
```

---

## Outputs

- **Streamlit dashboard** with per-page extraction, comparison charts, graph visualization
- **RDF/Turtle** knowledge graph (`.ttl`) — queryable with SPARQL
- **JSON** — full structured extraction per page
- **CSV** — flat entity table for downstream analysis

---

## Key Technical Concepts

### Cross-modal Alignment
The VLM receives both the rendered page image and the raw page text in a joint prompt, enabling it to ground visual elements (annotated map zones, chart legends) to their textual descriptions.

### Ontology Structure
Entities are typed using a lightweight domain ontology:
- `docv:Location`, `docv:Organization`, `docv:Species`
- `docv:Measurement`, `docv:RestorationMethod`, `docv:PhysicalObject`

Relations are encoded as RDF triples with provenance links back to source pages.

### Evaluation
`evaluator.py` computes:
- **Completeness gain** — % more entities extracted by VLM vs. text-only
- **Visual-only entity count** — entities only visible in images
- **Overlap** — % of text entities confirmed by multimodal extraction

---

## Example SPARQL Queries

```sparql
# All entities and their types
PREFIX docv: <http://docvision.io/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?label ?type WHERE {
    ?e a ?type ; rdfs:label ?label .
}

# Entities extracted only from visual modality
SELECT ?label WHERE {
    ?e rdfs:label ?label ;
       docv:extractionModality "visual" .
}
```

---

## Stack

| Component | Library |
|-----------|---------|
| PDF parsing | PyMuPDF (`fitz`) |
| VLM inference | HuggingFace Transformers (BLIP-2 / LLaVA) |
| Knowledge graph | RDFLib |
| UI | Streamlit |
| Visualization | Plotly |
| Data | Pandas |

---

## Extending the Project

- **Add ontology constraints**: plug in an OWL ontology via `owlready2` to validate extracted entities
- **Domain adaptation**: fine-tune the VLM on domain-specific document pairs
- **Multi-document graph**: merge RDF graphs across documents for cross-document querying
- **Evaluation benchmark**: annotate a gold-standard page and compute precision/recall per modality
