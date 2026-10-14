# tests/test_run_evals.py
#
# Phase 7: formalizes the sanity checks run ad hoc at the start of Phase 1
# (see docs/PROGRESS.md) into a repeatable test suite for the eval
# harness's own scoring logic -- the harness measures the pipeline, so bugs
# in the harness itself are exactly the kind of thing worth catching before
# trusting a number (this project found a real one this way: the "wifi" vs
# "Wi-Fi" tokenizer bug in Phase 1).

from run_evals import (
    tokenize, jaccard, score_entities, score_relationships,
    score_text_coverage, score_evidence_grounding,
    score_relationship_label_normalization,
)


# --- tokenize / jaccard -------------------------------------------------------

def test_tokenize_handles_hyphenated_compounds():
    # regression test for the Phase 1 "wifi" vs "Wi-Fi" bug: hyphens must be
    # stripped before splitting, so both forms produce the same token.
    assert tokenize("Wi-Fi outage") == tokenize("wifi outage")


def test_tokenize_crude_stemming_handles_simple_plurals():
    assert "nurse" in tokenize("nurses")


def test_jaccard_identical_sets():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_perfect_match():
    assert jaccard(set(), set()) == 1.0


# --- score_entities ------------------------------------------------------------

def test_score_entities_matches_on_fuzzy_label_with_type_gate():
    predicted = [
        {"id": "p1", "label": "patient", "type": "actor"},
        {"id": "p2", "label": "nurse", "type": "actor"},        # singular vs. plural GT
        {"id": "p3", "label": "Hallucinated Thing", "type": "component"},  # should not match anything
    ]
    gt = [
        {"label": "patient", "type": "actor"},
        {"label": "nurses", "type": "actor"},
    ]
    scores, pred_to_gt = score_entities(predicted, gt)
    assert scores["matched"] == 2
    assert scores["predicted_count"] == 3
    assert "p3" not in pred_to_gt


def test_score_entities_type_mismatch_does_not_match():
    predicted = [{"id": "p1", "label": "Mobile App", "type": "actor"}]  # wrong type
    gt = [{"label": "Mobile App", "type": "component"}]
    scores, _ = score_entities(predicted, gt)
    assert scores["matched"] == 0


def test_score_entities_empty_prediction_scores_zero_not_error():
    scores, pred_to_gt = score_entities([], [{"label": "X", "type": "actor"}])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


# --- score_relationships -------------------------------------------------------

def test_score_relationships_matches_on_entity_pair_ignoring_label_text():
    gt_entities = [{"label": "A", "type": "component"}, {"label": "B", "type": "component"}]
    pred_to_gt = {"pA": 0, "pB": 1}
    predicted_edges = [{"source": "pA", "target": "pB", "label": "any_label_text_at_all"}]
    gt_relationships = [{"source": "A", "target": "B"}]
    scores = score_relationships(predicted_edges, gt_relationships, gt_entities, pred_to_gt)
    assert scores["matched"] == 1
    assert scores["precision"] == 1.0


def test_score_relationships_unmatched_entity_endpoint_does_not_count():
    gt_entities = [{"label": "A", "type": "component"}, {"label": "B", "type": "component"}]
    pred_to_gt = {"pA": 0}  # pB never matched to any GT entity
    predicted_edges = [{"source": "pA", "target": "pB", "label": "x"}]
    gt_relationships = [{"source": "A", "target": "B"}]
    scores = score_relationships(predicted_edges, gt_relationships, gt_entities, pred_to_gt)
    assert scores["matched"] == 0


# --- score_text_coverage (risks / assumptions) --------------------------------

def test_score_text_coverage_recognizes_paraphrased_match_above_threshold():
    gt = ["network outage delays alerts"]
    predicted = ["An outage of the network may delay critical alerts to staff."]
    scores = score_text_coverage(predicted, gt)
    assert scores["recall"] == 1.0


def test_score_text_coverage_no_gt_items_is_vacuously_perfect():
    scores = score_text_coverage(["anything"], [])
    assert scores["recall"] == 1.0


# --- score_evidence_grounding (Phase 3) ---------------------------------------

def test_score_evidence_grounding_flags_fabricated_evidence():
    source = "A patient wears EEG/ECG sensors during monitoring."
    evidence = ["A patient wears EEG/ECG sensors during monitoring.", "This sentence does not appear anywhere.", ""]
    scores = score_evidence_grounding(evidence, source)
    assert scores["grounded"] == 1
    assert scores["total"] == 3


# --- score_relationship_label_normalization (Phase 3) -------------------------

def test_score_relationship_label_normalization_counts_canonical_labels():
    edges = [
        {"source": "a", "target": "b", "label": "sends_to"},       # canonical
        {"source": "b", "target": "c", "label": "some_weird_verb"},  # not canonical
    ]
    scores = score_relationship_label_normalization(edges)
    assert scores["normalized"] == 1
    assert scores["total"] == 2
