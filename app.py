"""
app.py
------
DocVision — Structured Information Extraction from Scientific PDFs
using Vision-Language Models (VLMs).

Run with:
    streamlit run app.py
"""

import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"  # fix torch.classes warning

import streamlit as st
import tempfile, json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="DocVision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:         #0D0F14;
    --surface:    #14171F;
    --surface2:   #1C2030;
    --border:     #2A2F3F;
    --accent:     #5B8FF9;
    --accent2:    #7EC8A4;
    --accent3:    #F4A45A;
    --text:       #E8EAF0;
    --text-muted: #8B91A8;
    --tag-bg:     #1E2535;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* Cards */
.dv-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

/* Entity tags */
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    margin: 2px 3px;
    font-weight: 500;
}
.tag-LOCATION     { background: #1A2D4A; color: #5B8FF9; border: 1px solid #2A4070; }
.tag-ORGANIZATION { background: #2A1A3A; color: #C07BF5; border: 1px solid #4A2A6A; }
.tag-SPECIES      { background: #1A3A2A; color: #7EC8A4; border: 1px solid #2A6040; }
.tag-MEASURE      { background: #3A2A1A; color: #F4A45A; border: 1px solid #6A4020; }
.tag-METHOD       { background: #1A3030; color: #5BC8C8; border: 1px solid #2A5050; }
.tag-OBJECT       { background: #2A2A1A; color: #C8C850; border: 1px solid #505020; }
.tag-visual       { background: #2A1A1A; color: #F47878; border: 1px solid #6A2A2A; }
.tag-both         { background: #1A2A1A; color: #78C878; border: 1px solid #2A6A2A; }

/* Metric boxes */
.metric-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
}
.hero h1 { font-size: 3.2rem; margin-bottom: 0.5rem; }
.hero .sub { color: var(--text-muted); font-size: 1.05rem; max-width: 600px; margin: 0 auto; }

/* SPARQL box */
.sparql-result {
    background: #0A0C10;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    overflow-x: auto;
}

/* Relation item */
.relation-row {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 0; border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
}
.rel-subj { color: var(--accent); font-weight: 500; }
.rel-pred { color: var(--text-muted); font-style: italic; }
.rel-obj  { color: var(--accent2); font-weight: 500; }

/* Scrollable image */
.page-image img { max-height: 400px; object-fit: contain; }

/* Upload zone */
.stFileUploader { border: 1px dashed var(--border); border-radius: 8px; }

/* Tab styling */
button[data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; }

/* Streamlit overrides */
.stButton > button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.4rem;
    font-size: 0.9rem;
}
.stButton > button:hover { opacity: 0.88; }

hr { border-color: var(--border); }

code { font-family: 'DM Mono', monospace; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Imports (after page config) ────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src import (
    pdf_to_pages, pil_to_base64,
    extract_text_only, extract_multimodal, merge_results,
    results_to_rdf, serialize_graph, run_sparql, SPARQL_QUERIES,
    compare_extractions,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper UI components
# ══════════════════════════════════════════════════════════════════════════════

def render_entity_tags(entities: list[dict]):
    html = ""
    for e in entities:
        etype    = e.get("type", "OBJECT")
        modality = e.get("modality", e.get("source", "text"))
        css      = f"tag tag-{etype}"
        label    = f"{e['text']} [{etype}]"
        if modality in ("visual", "both"):
            css += f" tag-{modality}"
        html += f'<span class="{css}">{label}</span> '
    st.markdown(html, unsafe_allow_html=True)


def render_relations(relations: list[dict]):
    if not relations:
        st.markdown("*No relations extracted.*")
        return
    html = ""
    for r in relations:
        src_icon = "👁️" if r.get("modality") == "visual" else ("🔗" if r.get("modality") == "both" else "📄")
        html += (
            f'<div class="relation-row">{src_icon} '
            f'<span class="rel-subj">{r["subject"]}</span> '
            f'<span class="rel-pred">—[{r["predicate"]}]→</span> '
            f'<span class="rel-obj">{r["object"]}</span>'
            f'</div>'
        )
    st.markdown(html, unsafe_allow_html=True)


def metric_box(value, label):
    return f"""
    <div class="metric-box">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


def make_entity_bar_chart(text_types: dict, mm_types: dict) -> go.Figure:
    all_types = sorted(set(list(text_types.keys()) + list(mm_types.keys())))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Text-only", x=all_types,
        y=[text_types.get(t, 0) for t in all_types],
        marker_color="#5B8FF9",
    ))
    fig.add_trace(go.Bar(
        name="Multimodal", x=all_types,
        y=[mm_types.get(t, 0) for t in all_types],
        marker_color="#7EC8A4",
    ))
    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans", margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=-0.25),
        height=260,
    )
    return fig


def make_graph_network(relations: list[dict]) -> go.Figure:
    """Simple force-style graph using scatter + lines."""
    import math, random
    nodes = {}
    edges = []
    for r in relations:
        for n in (r["subject"], r["object"]):
            if n not in nodes:
                nodes[n] = len(nodes)
        edges.append((r["subject"], r["object"], r["predicate"]))

    n = len(nodes)
    if n == 0:
        return go.Figure()

    # Circular layout
    angles = {name: 2 * math.pi * i / n for i, name in enumerate(nodes)}
    pos = {name: (math.cos(a), math.sin(a)) for name, a in angles.items()}

    edge_x, edge_y = [], []
    for s, o, _ in edges:
        x0, y0 = pos[s]; x1, y1 = pos[o]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_labels = list(nodes.keys())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
        line=dict(color="#2A2F3F", width=1.5), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="top center",
        marker=dict(size=14, color="#5B8FF9", line=dict(color="#1C2030", width=2)),
        textfont=dict(size=9, color="#E8EAF0"),
    ))
    fig.update_layout(
        showlegend=False, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=0, b=0), height=380,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Session state
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    for key, val in [
        ("pages",           None),
        ("text_results",    None),
        ("mm_results",      None),
        ("merged_results",  None),
        ("rdf_graph",       None),
        ("comparison",      None),
        ("pdf_name",        ""),
        ("processed",       False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

init_state()


# ══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    backend = st.selectbox(
        "VLM Backend",
        options=["mock", "blip2", "llava"],
        help=(
            "**mock** — instant demo, no GPU needed.\n\n"
            "**blip2** — Salesforce/blip2-opt-2.7b (~6 GB VRAM).\n\n"
            "**llava** — llava-hf/llava-1.5-7b-hf (~14 GB VRAM)."
        ),
    )

    dpi = st.slider("PDF render DPI", 72, 300, 150, 12,
                    help="Higher DPI = better image quality but slower processing.")

    max_pages = st.slider("Max pages to process", 1, 20, 5,
                          help="Limit pages for faster runs during development.")

    st.divider()
    st.markdown("### 📊 Modality Legend")
    st.markdown("""
<span class="tag tag-LOCATION">LOCATION</span>
<span class="tag tag-ORGANIZATION">ORGANIZATION</span>
<span class="tag tag-SPECIES">SPECIES</span>
<span class="tag tag-MEASURE">MEASURE</span>
<span class="tag tag-METHOD">METHOD</span>
<span class="tag tag-OBJECT">OBJECT</span>

<br><br>

<span class="tag tag-visual">👁️ visual-only</span>
<span class="tag tag-both">🔗 both modalities</span>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<span style='color:#8B91A8;font-size:0.78rem'>"
        "DocVision · VLM-based multimodal extraction<br>"
        "Stack: PyMuPDF · HuggingFace · RDFLib · Streamlit"
        "</span>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Hero + Upload
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <h1>Doc<span style="color:#5B8FF9">Vision</span></h1>
    <p class="sub">
        Structured information extraction from scientific PDFs —<br>
        comparing <em>text-only</em> vs. <em>Vision-Language Model</em> multimodal pipelines.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a scientific or technical PDF here",
    type=["pdf"],
    label_visibility="collapsed",
)

if uploaded:
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run = st.button("🚀 Run Extraction", use_container_width=True)
    with col_info:
        st.markdown(
            f"<span style='color:#8B91A8;font-size:0.85rem'>📄 {uploaded.name} · "
            f"backend: <code>{backend}</code> · DPI: {dpi} · max pages: {max_pages}</span>",
            unsafe_allow_html=True,
        )

    if run:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Parsing PDF pages…"):
            pages = pdf_to_pages(tmp_path, dpi=dpi)[:max_pages]
            st.session_state.pages = pages
            st.session_state.pdf_name = uploaded.name

        text_results = []
        mm_results   = []
        merged       = []

        progress = st.progress(0, text="Extracting…")
        for i, page in enumerate(pages):
            progress.progress((i) / len(pages), text=f"Page {i+1}/{len(pages)}")

            # Text-only
            tr = extract_text_only(page.raw_text)
            text_results.append(tr)

            # Multimodal (VLM)
            mr = extract_multimodal(page.page_image, page.raw_text, backend=backend)
            mm_results.append(mr)

            # Merged
            merged.append(merge_results(tr, mr))

        progress.progress(1.0, text="Building knowledge graph…")

        rdf_g = results_to_rdf(merged, source_label=uploaded.name)

        def fold_results(results):
            acc = results[0]
            for r in results[1:]:
                acc = merge_results(acc, r)
            return acc

        comp = compare_extractions(
            fold_results(text_results),
            fold_results(mm_results),
        ) if text_results else None

        st.session_state.text_results   = text_results
        st.session_state.mm_results     = mm_results
        st.session_state.merged_results = merged
        st.session_state.rdf_graph      = rdf_g
        st.session_state.comparison     = comp
        st.session_state.processed      = True

        progress.empty()
        os.unlink(tmp_path)
        st.success(f"✅ Processed {len(pages)} page(s) from **{uploaded.name}**")


# ══════════════════════════════════════════════════════════════════════════════
#  Results
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.processed:
    pages        = st.session_state.pages
    text_results = st.session_state.text_results
    mm_results   = st.session_state.mm_results
    merged       = st.session_state.merged_results
    rdf_g        = st.session_state.rdf_graph
    comp         = st.session_state.comparison

    # ── Top-level metrics ──────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    total_text_ent = sum(len(r.entities) for r in text_results)
    total_mm_ent   = sum(len(r.entities) for r in mm_results)
    total_rels     = sum(len(r.relations) for r in merged)
    total_triples  = len(rdf_g)

    c1.markdown(metric_box(len(pages), "Pages"), unsafe_allow_html=True)
    c2.markdown(metric_box(total_text_ent, "Text Entities"), unsafe_allow_html=True)
    c3.markdown(metric_box(total_mm_ent,   "VLM Entities"),  unsafe_allow_html=True)
    c4.markdown(metric_box(total_rels,     "Relations"),     unsafe_allow_html=True)
    c5.markdown(metric_box(total_triples,  "RDF Triples"),   unsafe_allow_html=True)

    st.markdown("---")

    # ── Tabs ───────────────────────────────────────────────────────────────
    tab_pages, tab_compare, tab_graph, tab_kg, tab_sparql, tab_export = st.tabs([
        "📄 Pages", "📊 Comparison", "🕸️ Graph", "🗂️ Knowledge Graph", "🔍 SPARQL", "💾 Export"
    ])

    # ── TAB: Pages ─────────────────────────────────────────────────────────
    with tab_pages:
        for i, (page, tr, mr) in enumerate(zip(pages, text_results, mm_results)):
            with st.expander(f"Page {page.page_number}", expanded=(i == 0)):
                left, right = st.columns([1, 1])

                with left:
                    st.markdown("**Page render**")
                    b64 = pil_to_base64(page.page_image)
                    st.markdown(
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="width:100%;border-radius:8px;border:1px solid #2A2F3F"/>',
                        unsafe_allow_html=True,
                    )
                    if page.raw_text:
                        if st.checkbox("Show raw text", key=f"raw_{i}"):
                            st.text(page.raw_text[:1500] + ("…" if len(page.raw_text) > 1500 else ""))

                with right:
                    st.markdown("**📄 Text-only extraction**")
                    render_entity_tags(tr.entities)
                    st.markdown("*Relations:*")
                    render_relations(tr.relations[:8])

                    st.divider()

                    st.markdown("**🔬 Multimodal extraction (VLM)**")
                    render_entity_tags(mr.entities)
                    st.markdown("*Relations:*")
                    render_relations(mr.relations[:8])

                    if mr.summary:
                        st.markdown("*Summary:*")
                        st.markdown(
                            f'<div class="dv-card" style="margin-top:8px">'
                            f'<span style="font-size:0.87rem;color:#C8CAD8">{mr.summary}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── TAB: Comparison ────────────────────────────────────────────────────
    with tab_compare:
        if comp:
            st.markdown("### Text-only vs. Multimodal — Completeness Analysis")
            col1, col2, col3 = st.columns(3)
            col1.markdown(
                metric_box(f"+{comp.completeness_gain_pct}%", "Entity Completeness Gain"),
                unsafe_allow_html=True,
            )
            col2.markdown(
                metric_box(comp.visual_only_entity_count, "Visual-only Entities"),
                unsafe_allow_html=True,
            )
            col3.markdown(
                metric_box(comp.visual_only_relation_count, "Visual-only Relations"),
                unsafe_allow_html=True,
            )

            st.markdown("#### Entity type distribution")
            st.plotly_chart(
                make_entity_bar_chart(comp.text_entity_types, comp.multimodal_entity_types),
                use_container_width=True,
            )

            if comp.new_entity_examples:
                st.markdown("#### Entities found only by the VLM (visual-only)")
                tags = "".join(
                    f'<span class="tag tag-visual">👁️ {e}</span> '
                    for e in comp.new_entity_examples
                )
                st.markdown(tags, unsafe_allow_html=True)

            # Summary table
            df = pd.DataFrame({
                "Metric":     ["Entities", "Relations", "Overlap (%)"],
                "Text-only":  [str(comp.text_entity_count),
                                str(comp.relation_text_count),
                                f"{comp.overlap_pct}%"],
                "Multimodal": [str(comp.multimodal_entity_count),
                                str(comp.relation_multimodal_count),
                                "—"],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── TAB: Graph ─────────────────────────────────────────────────────────
    with tab_graph:
        all_rels = [r for res in merged for r in res.relations]
        if all_rels:
            st.markdown("### Extracted Knowledge Network")
            st.plotly_chart(make_graph_network(all_rels[:40]), use_container_width=True)
            st.caption("Showing up to 40 relations. Nodes = entities, edges = relations.")
        else:
            st.info("No relations extracted yet.")

    # ── TAB: Knowledge Graph ───────────────────────────────────────────────
    with tab_kg:
        st.markdown("### RDF Knowledge Graph")
        all_ents  = [e for res in merged for e in res.entities]
        all_rels2 = [r for res in merged for r in res.relations]

        col_e, col_r = st.columns(2)
        with col_e:
            st.markdown("**All Entities**")
            if all_ents:
                df_e = pd.DataFrame(all_ents)
                st.dataframe(df_e[["text","type","modality"] if "modality" in df_e.columns else ["text","type"]],
                             use_container_width=True, hide_index=True)
        with col_r:
            st.markdown("**All Relations**")
            if all_rels2:
                df_r = pd.DataFrame(all_rels2)
                cols = [c for c in ["subject","predicate","object","modality"] if c in df_r.columns]
                st.dataframe(df_r[cols], use_container_width=True, hide_index=True)

        st.markdown("**RDF Turtle preview**")
        ttl = serialize_graph(rdf_g, fmt="turtle")
        st.code(ttl[:3000] + ("\n…" if len(ttl) > 3000 else ""), language="turtle")

    # ── TAB: SPARQL ─────────────────────────────────────────────────────────
    with tab_sparql:
        st.markdown("### SPARQL Query Interface")

        preset = st.selectbox("Preset queries", list(SPARQL_QUERIES.keys()))
        query  = st.text_area("SPARQL Query", value=SPARQL_QUERIES[preset], height=200)

        if st.button("▶ Run Query"):
            rows = run_sparql(rdf_g, query)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Query returned no results.")

    # ── TAB: Export ─────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### Export Extraction Results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**RDF / Turtle**")
            ttl = serialize_graph(rdf_g, "turtle")
            st.download_button("⬇ Download .ttl", ttl, file_name="docvision_kg.ttl",
                               mime="text/turtle", use_container_width=True)

        with col_b:
            st.markdown("**JSON — Merged Results**")
            export_json = json.dumps(
                [{"page": i+1, "entities": r.entities, "relations": r.relations, "summary": r.summary}
                 for i, r in enumerate(merged)],
                indent=2, ensure_ascii=False,
            )
            st.download_button("⬇ Download .json", export_json, file_name="docvision_results.json",
                               mime="application/json", use_container_width=True)

        with col_c:
            st.markdown("**Entities CSV**")
            all_ents_flat = [
                {"page": i+1, **e}
                for i, res in enumerate(merged)
                for e in res.entities
            ]
            if all_ents_flat:
                csv = pd.DataFrame(all_ents_flat).to_csv(index=False)
                st.download_button("⬇ Download .csv", csv, file_name="docvision_entities.csv",
                                   mime="text/csv", use_container_width=True)


# ── Empty state ─────────────────────────────────────────────────────────────
if not uploaded:
    st.markdown("""
<div style="text-align:center; padding:4rem 2rem; color:#8B91A8">
    <div style="font-size:3rem">📄</div>
    <p style="margin-top:1rem">Upload a PDF to begin extraction</p>
    <p style="font-size:0.82rem;max-width:480px;margin:0 auto">
        Supports scientific reports, technical documents, environmental studies, and research papers.
        Switch to <strong>mock</strong> backend for instant demo without a GPU.
    </p>
</div>
""", unsafe_allow_html=True)