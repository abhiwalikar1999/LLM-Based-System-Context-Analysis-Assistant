# tests/test_core.py
#
# Phase 7: unit tests for core.py's pure-Python graph logic. No network
# calls anywhere in this file -- build_context_graph is tested via
# monkeypatching llm_engine.ask_chatgpt, the same pattern used informally
# throughout this project's earlier phases (see docs/PROGRESS.md), now
# formalized as a real, repeatable, CI-able test instead of an ad-hoc
# one-off check.

import pytest

import core
from core import (
    ContextGraph, _norm, _safe_id, ensure_node, add_edge, dedupe_edges,
    normalize_relationship_label, canonicalize_extraction,
    apply_base_improvements, extract_prefixed_lines,
    extract_functional_requirements_from_text, build_context_graph,
)


# --- _norm -----------------------------------------------------------------

def test_norm_lowercases_and_collapses_whitespace():
    assert _norm("  Patient   Monitoring   Device ") == "patient monitoring device"


def test_norm_handles_none_and_empty():
    assert _norm(None) == ""
    assert _norm("") == ""


# --- _safe_id ---------------------------------------------------------------

def test_safe_id_is_deterministic():
    assert _safe_id("Mobile App", "component") == _safe_id("Mobile App", "component")


def test_safe_id_is_case_and_whitespace_insensitive():
    assert _safe_id("Mobile App", "component") == _safe_id("  mobile   app  ", "component")


def test_safe_id_differs_by_type():
    assert _safe_id("X", "actor") != _safe_id("X", "component")


# --- ensure_node (dedup logic) ----------------------------------------------

def test_ensure_node_creates_new_node():
    graph = ContextGraph()
    nid = ensure_node(graph, "Mobile App", "component")
    assert nid in graph.nodes
    assert graph.nodes[nid].label == "Mobile App"
    assert graph.nodes[nid].type == "component"


def test_ensure_node_dedups_by_normalized_label_and_type():
    graph = ContextGraph()
    nid1 = ensure_node(graph, "Mobile App", "component")
    nid2 = ensure_node(graph, "  mobile   app  ", "component")  # same entity, different casing/whitespace
    assert nid1 == nid2
    assert len(graph.nodes) == 1


def test_ensure_node_dedup_is_punctuation_sensitive_not_just_whitespace_and_case():
    # Discovered while writing this test suite (not assumed beforehand):
    # _norm() only lowercases and collapses whitespace -- it does NOT strip
    # punctuation. So "EEG/ECG sensors" and "EEG ECG sensors" normalize to
    # two DIFFERENT strings ("eeg/ecg sensors" vs "eeg ecg sensors") and do
    # NOT dedupe, even though a human reader would consider them the same
    # entity. This is exactly why run_evals.py's own entity matching uses
    # fuzzy token-Jaccard similarity instead of relying on this exact-match
    # dedup (see run_evals.py's header comment) -- this test documents that
    # the production dedup logic is deliberately simple/exact, not fuzzy.
    graph = ContextGraph()
    nid1 = ensure_node(graph, "EEG/ECG sensors", "component")
    nid2 = ensure_node(graph, "EEG ECG sensors", "component")
    assert nid1 != nid2
    assert len(graph.nodes) == 2


def test_ensure_node_same_label_different_type_creates_separate_nodes():
    graph = ContextGraph()
    nid_actor = ensure_node(graph, "Admin", "actor")
    nid_component = ensure_node(graph, "Admin", "component")
    assert nid_actor != nid_component
    assert len(graph.nodes) == 2


def test_ensure_node_preserves_evidence_only_on_first_creation():
    graph = ContextGraph()
    nid = ensure_node(graph, "Mobile App", "component", evidence="first evidence")
    ensure_node(graph, "Mobile App", "component", evidence="second evidence, should be ignored")
    assert graph.nodes[nid].evidence == "first evidence"


# --- add_edge ----------------------------------------------------------------

def test_add_edge_between_existing_nodes():
    graph = ContextGraph()
    a = ensure_node(graph, "A", "component")
    b = ensure_node(graph, "B", "component")
    add_edge(graph, a, b, "calls")
    assert len(graph.edges) == 1
    assert graph.edges[0].source == a and graph.edges[0].target == b


def test_add_edge_silently_skipped_when_endpoint_unknown():
    graph = ContextGraph()
    a = ensure_node(graph, "A", "component")
    add_edge(graph, a, "nonexistent-id", "calls")
    assert len(graph.edges) == 0


# --- dedupe_edges --------------------------------------------------------------

def test_dedupe_edges_removes_exact_duplicates():
    graph = ContextGraph()
    a = ensure_node(graph, "A", "component")
    b = ensure_node(graph, "B", "component")
    add_edge(graph, a, b, "calls")
    add_edge(graph, a, b, "calls")  # exact duplicate
    add_edge(graph, a, b, "notifies")  # different label, not a duplicate
    deduped = dedupe_edges(graph.edges)
    assert len(deduped) == 2


# --- normalize_relationship_label (Phase 3) ---------------------------------

def test_normalize_relationship_label_maps_known_variants_to_canonical_verb():
    assert normalize_relationship_label("streams") == "sends_to"
    assert normalize_relationship_label("sends_signals_to") == "sends_to"
    assert normalize_relationship_label("provides_alerts_to") == "sends_to"


def test_normalize_relationship_label_passes_through_unmapped_label_unchanged():
    assert normalize_relationship_label("some_totally_novel_verb") == "some_totally_novel_verb"


def test_normalize_relationship_label_is_case_insensitive():
    assert normalize_relationship_label("STREAMS") == "sends_to"


# --- canonicalize_extraction (graph building from a fixed fake LLM response) -

FAKE_EXTRACTION = {
    "entities": [
        {"id": "e1", "label": "Patient", "type": "actor", "evidence": "A patient wears sensors."},
        {"id": "e2", "label": "EEG/ECG sensors", "type": "component", "evidence": "A patient wears sensors."},
        # same real-world entity as e2: _norm only collapses case/whitespace,
        # not punctuation, so this must match on case+whitespace to actually
        # dedupe -- changing the punctuation (e.g. "EEG ECG") would NOT merge,
        # which is itself worth knowing about ensure_node's real behavior.
        {"id": "e3", "label": "  eeg/ecg   SENSORS  ", "type": "component", "evidence": "duplicate of e2, different casing/spacing"},
    ],
    "relationships": [
        {"source_id": "e1", "target_id": "e2", "label": "wears", "evidence": "A patient wears sensors."},
        # references the near-duplicate id (e3), which should resolve to the same canonical node as e2
        {"source_id": "e3", "target_id": "e1", "label": "monitors", "evidence": "ev"},
    ],
    "risks": [
        {"id": "r1", "description": "sensor disconnects", "severity": "medium", "related_nodes": ["e2"], "evidence": "ev"},
    ],
    "assumptions": [
        {"id": "a1", "description": "sensors stay attached", "related_nodes": ["e2"], "evidence": "ev"},
    ],
}


def test_canonicalize_extraction_builds_expected_graph_shape():
    graph = canonicalize_extraction(FAKE_EXTRACTION)
    # e2 and e3 are the same real-world entity (near-duplicate label), so they
    # must canonicalize to ONE node, not two.
    assert len(graph.nodes) == 2
    assert len(graph.risks) == 1
    assert len(graph.assumptions) == 1


def test_canonicalize_extraction_rewrites_relationship_endpoints_to_canonical_ids():
    graph = canonicalize_extraction(FAKE_EXTRACTION)
    sensor_node_id = next(nid for nid, n in graph.nodes.items() if n.type == "component")
    patient_node_id = next(nid for nid, n in graph.nodes.items() if n.type == "actor")
    edge_pairs = {(e.source, e.target) for e in graph.edges}
    # both relationships (one referencing e2 directly, one referencing the
    # near-duplicate e3) must resolve to the SAME canonical sensor node id
    assert (patient_node_id, sensor_node_id) in edge_pairs
    assert (sensor_node_id, patient_node_id) in edge_pairs


def test_canonicalize_extraction_rewrites_risk_and_assumption_related_nodes():
    graph = canonicalize_extraction(FAKE_EXTRACTION)
    sensor_node_id = next(nid for nid, n in graph.nodes.items() if n.type == "component")
    assert graph.risks[0].related_nodes == [sensor_node_id]
    assert graph.assumptions[0].related_nodes == [sensor_node_id]


def test_canonicalize_extraction_preserves_evidence():
    graph = canonicalize_extraction(FAKE_EXTRACTION)
    sensor_node = next(n for n in graph.nodes.values() if n.type == "component")
    assert sensor_node.evidence == "A patient wears sensors."


def test_canonicalize_extraction_applies_relationship_normalization():
    result = {
        "entities": [
            {"id": "e1", "label": "A", "type": "component", "evidence": ""},
            {"id": "e2", "label": "B", "type": "component", "evidence": ""},
        ],
        "relationships": [{"source_id": "e1", "target_id": "e2", "label": "streams", "evidence": ""}],
        "risks": [], "assumptions": [],
    }
    graph = canonicalize_extraction(result)
    assert graph.edges[0].label == "sends_to"


def test_canonicalize_extraction_handles_missing_optional_fields_gracefully():
    # a minimal, schema-shaped dict with no evidence/related_nodes keys at all
    result = {
        "entities": [{"id": "e1", "label": "A", "type": "component"}],
        "relationships": [],
        "risks": [{"id": "r1", "description": "x", "severity": "low"}],
        "assumptions": [{"id": "a1", "description": "y"}],
    }
    graph = canonicalize_extraction(result)
    assert len(graph.nodes) == 1
    assert graph.risks[0].related_nodes == []
    assert graph.assumptions[0].related_nodes == []


# --- apply_base_improvements (deterministic skeleton, Phase 0) -------------

def test_apply_base_improvements_adds_expected_node_counts():
    graph = ContextGraph()
    apply_base_improvements(graph)
    actors = [n for n in graph.nodes.values() if n.type == "actor"]
    components = [n for n in graph.nodes.values() if n.type == "component"]
    external_systems = [n for n in graph.nodes.values() if n.type == "external_system"]
    assert len(actors) == 4       # patient, nurses, doctors, IT administrators
    assert len(components) == 8   # sensors, device, gateway, analytics, dashboard, notify, data_store, audit
    assert len(external_systems) == 2  # EHR, Hospital Identity Provider


def test_apply_base_improvements_is_idempotent():
    graph = ContextGraph()
    apply_base_improvements(graph)
    node_count_after_first = len(graph.nodes)
    edge_count_after_first = len(graph.edges)
    apply_base_improvements(graph)  # calling it again should not duplicate anything
    assert len(graph.nodes) == node_count_after_first
    assert len(graph.edges) == edge_count_after_first


def test_apply_base_improvements_merges_with_existing_llm_extracted_node():
    graph = ContextGraph()
    # simulate the LLM having already extracted the same real-world entity
    # under a slightly different label before the deterministic layer runs
    llm_sensor_id = ensure_node(graph, "eeg/ecg sensors", "component", evidence="from the LLM")
    apply_base_improvements(graph)
    sensor_nodes = [n for n in graph.nodes.values() if "sensor" in n.label.lower()]
    assert len(sensor_nodes) == 1  # must merge, not create a second sensor node
    assert graph.nodes[llm_sensor_id].evidence == "from the LLM"  # original evidence preserved


# --- extract_prefixed_lines / extract_functional_requirements_from_text ----

def test_extract_prefixed_lines_finds_matching_lines():
    text = "Some intro.\nRisk: Network outage.\nAssumption: Stable power.\nRisk: Sensor failure."
    assert extract_prefixed_lines(text, "Risk") == ["Network outage.", "Sensor failure."]
    assert extract_prefixed_lines(text, "Assumption") == ["Stable power."]


def test_extract_prefixed_lines_is_case_insensitive_on_prefix():
    text = "RISK: something bad."
    assert extract_prefixed_lines(text, "Risk") == ["something bad."]


def test_extract_prefixed_lines_empty_when_no_match():
    assert extract_prefixed_lines("Nothing relevant here.", "Risk") == []


def test_extract_functional_requirements_matches_known_keywords():
    frs = extract_functional_requirements_from_text("The system uses EEG and ECG sensors with a dashboard.")
    assert any(fr.startswith("FR1:") for fr in frs)  # EEG/ECG keyword
    assert any(fr.startswith("FR6:") for fr in frs)  # dashboard keyword


def test_extract_functional_requirements_falls_back_when_no_keywords_match():
    frs = extract_functional_requirements_from_text("A completely unrelated piece of text about gardening.")
    assert len(frs) == 4  # the generic fallback list
    assert frs[0].startswith("FR1: Acquire physiological signals")


def test_extract_functional_requirements_preserves_order_and_dedupes():
    frs = extract_functional_requirements_from_text("dashboard dashboard dashboard")
    # "dashboard" only triggers FR6 once, even though it appears 3 times
    assert frs.count(next(fr for fr in frs if fr.startswith("FR6:"))) == 1


# --- build_context_graph with a mocked LLM call (no network) ---------------

def test_build_context_graph_with_single_string_input(monkeypatch):
    monkeypatch.setattr(core, "ask_chatgpt", lambda text, model=None: dict(FAKE_EXTRACTION))

    result, graph = build_context_graph("A patient wears EEG/ECG sensors.")
    assert len(graph.nodes) == 2  # e2/e3 dedup, same as the canonicalize_extraction test above
    assert "_retrieval_stats" in result
    assert result["_retrieval_stats"]["retrieval_activated"] is False  # short input, below threshold


def test_build_context_graph_passes_model_override_through(monkeypatch):
    captured = {}

    def fake_ask_chatgpt(text, model=None):
        captured["model"] = model
        return dict(FAKE_EXTRACTION)

    monkeypatch.setattr(core, "ask_chatgpt", fake_ask_chatgpt)
    build_context_graph("some text", model="some-other-model")
    assert captured["model"] == "some-other-model"


def test_build_context_graph_default_call_does_not_override_model(monkeypatch):
    captured = {}

    def fake_ask_chatgpt(text, model=None):
        captured["model"] = model
        return dict(FAKE_EXTRACTION)

    monkeypatch.setattr(core, "ask_chatgpt", fake_ask_chatgpt)
    build_context_graph("some text")  # no model= passed
    assert captured["model"] is None  # confirms core.py didn't pass an override
