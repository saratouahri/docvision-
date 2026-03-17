

from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD
from rdflib.namespace import SKOS
import re
from .text_extractor import ExtractionResult


# ── Namespaces ─────────────────────────────────────────────────────────────────
DOCV  = Namespace("http://docvision.io/ontology#")
DOCVI = Namespace("http://docvision.io/instance/")
PROV  = Namespace("http://www.w3.org/ns/prov#")

# Map entity types to ontology classes
TYPE_MAP = {
    "LOCATION":     DOCV.Location,
    "ORGANIZATION": DOCV.Organization,
    "SPECIES":      DOCV.Species,
    "MEASURE":      DOCV.Measurement,
    "METHOD":       DOCV.RestorationMethod,
    "OBJECT":       DOCV.PhysicalObject,
}

# Map relation predicates to URIs
PRED_MAP = {
    "restoredAt":   DOCV.restoredAt,
    "locatedIn":    DOCV.locatedIn,
    "implemented":  DOCV.implemented,
    "measuredAs":   DOCV.measuredAs,
    "foundIn":      DOCV.foundIn,
    "covers":       DOCV.covers,
    "partOf":       DOCV.partOf,
    "relatedTo":    DOCV.relatedTo,
}


def _slugify(text: str) -> str:
    """Convert a string to a URI-safe slug."""
    slug = re.sub(r'[^a-zA-Z0-9_]', '_', text.strip())
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug[:80]  # cap length


def build_ontology(g: Graph) -> Graph:
    """Add basic ontology definitions to the graph."""
    g.bind("docv",  DOCV)
    g.bind("docvi", DOCVI)
    g.bind("prov",  PROV)
    g.bind("skos",  SKOS)

    classes = [
        (DOCV.Entity,            "Top-level entity"),
        (DOCV.Location,          "A geographic or spatial location"),
        (DOCV.Organization,      "An organization or institution"),
        (DOCV.Species,           "A biological species"),
        (DOCV.Measurement,       "A quantitative measurement"),
        (DOCV.RestorationMethod, "A restoration intervention method"),
        (DOCV.PhysicalObject,    "A physical or engineered object"),
        (DOCV.DocumentPage,      "A page from a source document"),
    ]
    for cls, label in classes:
        g.add((cls, RDF.type,    OWL.Class))
        g.add((cls, RDFS.label,  Literal(label)))
        if cls != DOCV.Entity:
            g.add((cls, RDFS.subClassOf, DOCV.Entity))

    properties = [
        (DOCV.restoredAt,        "restored at",         DOCV.Location),
        (DOCV.locatedIn,         "located in",          DOCV.Location),
        (DOCV.implemented,       "implemented",         DOCV.RestorationMethod),
        (DOCV.measuredAs,        "measured as",         DOCV.Measurement),
        (DOCV.foundIn,           "found in",            DOCV.Location),
        (DOCV.covers,            "covers area",         DOCV.Measurement),
        (DOCV.partOf,            "part of",             DOCV.Entity),
        (DOCV.relatedTo,         "related to",          DOCV.Entity),
        (DOCV.extractedFrom,     "extracted from",      DOCV.DocumentPage),
        (DOCV.extractionModality,"extraction modality", None),
    ]
    for prop, label, rng in properties:
        g.add((prop, RDF.type,   OWL.ObjectProperty))
        g.add((prop, RDFS.label, Literal(label)))
        if rng:
            g.add((prop, RDFS.range, rng))

    return g


def results_to_rdf(
    results: list[ExtractionResult],
    source_label: str = "document",
) -> Graph:
    """
    Convert a list of ExtractionResult objects into an RDF graph.

    Args:
        results:      One ExtractionResult per page (can mix text-only and multimodal).
        source_label: Human-readable label for the source document.

    Returns:
        rdflib.Graph with all triples.
    """
    g = Graph()
    build_ontology(g)

    doc_uri = DOCVI[_slugify(source_label)]
    g.add((doc_uri, RDF.type,   DOCV.Document))
    g.add((doc_uri, RDFS.label, Literal(source_label)))

    entity_uri_cache: dict[str, URIRef] = {}

    def get_entity_uri(text: str, etype: str) -> URIRef:
        key = text.lower()
        if key not in entity_uri_cache:
            uri = DOCVI[_slugify(text)]
            entity_uri_cache[key] = uri
            cls = TYPE_MAP.get(etype, DOCV.Entity)
            g.add((uri, RDF.type,   cls))
            g.add((uri, RDFS.label, Literal(text)))
        return entity_uri_cache[key]

    for idx, result in enumerate(results):
        page_uri = DOCVI[f"{_slugify(source_label)}_page{idx+1}"]
        g.add((page_uri, RDF.type,        DOCV.DocumentPage))
        g.add((page_uri, DOCV.partOf,     doc_uri))
        g.add((page_uri, RDFS.label,      Literal(f"Page {idx+1}")))
        g.add((page_uri, DOCV.extractionModality, Literal(result.modality)))

        # Entities
        for ent in result.entities:
            euri = get_entity_uri(ent["text"], ent.get("type", "OBJECT"))
            g.add((euri, DOCV.extractedFrom, page_uri))
            if ent.get("modality"):
                g.add((euri, DOCV.extractionModality, Literal(ent["modality"])))

        # Relations
        for rel in result.relations:
            subj_uri = get_entity_uri(rel["subject"],  "OBJECT")
            obj_uri  = get_entity_uri(rel["object"],   "OBJECT")
            pred     = PRED_MAP.get(rel["predicate"], DOCV.relatedTo)
            g.add((subj_uri, pred, obj_uri))
            # Provenance annotation
            stmt_uri = DOCVI[f"stmt_{_slugify(rel['subject'])}_{rel['predicate']}_{_slugify(rel['object'])}"]
            g.add((stmt_uri, RDF.type,            PROV.Entity))
            g.add((stmt_uri, PROV.wasDerivedFrom, page_uri))
            g.add((stmt_uri, DOCV.extractionModality, Literal(rel.get("modality", result.modality))))

    return g


def serialize_graph(g: Graph, fmt: str = "turtle") -> str:
    """Serialize the graph to a string in the given format (turtle, n3, xml, json-ld)."""
    return g.serialize(format=fmt)


SPARQL_QUERIES = {
    "All entities": """
PREFIX docv: <http://docvision.io/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?label ?type WHERE {
    ?e a ?type ;
       rdfs:label ?label .
    FILTER(?type != docv:DocumentPage && ?type != docv:Document)
}
ORDER BY ?type ?label
""",
    "All relations": """
PREFIX docv: <http://docvision.io/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?subject ?predicate ?object WHERE {
    ?s rdfs:label ?subject ;
       ?pred ?o .
    ?o rdfs:label ?object .
    ?pred rdfs:label ?predicate .
    FILTER(?pred != <http://www.w3.org/1999/02/22-rdf-syntax-ns#type>)
    FILTER(?pred != <http://www.w3.org/2000/01/rdf-schema#label>)
    FILTER(STRSTARTS(STR(?pred), "http://docvision.io/ontology#"))
}
""",
    "Visual-only entities": """
PREFIX docv: <http://docvision.io/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?label ?type WHERE {
    ?e a ?type ;
       rdfs:label ?label ;
       docv:extractionModality "visual" .
}
""",
    "Locations": """
PREFIX docv: <http://docvision.io/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?label WHERE {
    ?e a docv:Location ;
       rdfs:label ?label .
}
ORDER BY ?label
""",
}


def run_sparql(g: Graph, query: str) -> list[dict]:
    """Run a SPARQL SELECT query and return results as a list of dicts."""
    rows = []
    try:
        qres = g.query(query)
        for row in qres:
            rows.append({str(var): str(val) for var, val in zip(qres.vars, row)})
    except Exception as e:
        rows.append({"error": str(e)})
    return rows
