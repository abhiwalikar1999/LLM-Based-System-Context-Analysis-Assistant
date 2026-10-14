# tests/test_llm_engine.py
#
# Phase 7: formalizes the Phase 2 referential-integrity validator checks.
# No network: ContextExtraction's model_validator runs entirely locally.

import pytest
from pydantic import ValidationError

from llm_engine import ContextExtraction


VALID = {
    "entities": [
        {"id": "a", "label": "A", "type": "component"},
        {"id": "b", "label": "B", "type": "component"},
    ],
    "relationships": [{"source_id": "a", "target_id": "b", "label": "calls"}],
    "risks": [{"id": "r1", "description": "x", "severity": "low", "related_nodes": ["a"]}],
    "assumptions": [],
}


def test_valid_extraction_passes_validation():
    extraction = ContextExtraction(**VALID)
    assert extraction.entities[0].id == "a"


def test_dangling_relationship_target_is_rejected():
    bad = {**VALID, "relationships": [{"source_id": "a", "target_id": "ghost", "label": "calls"}]}
    with pytest.raises(ValidationError, match="ghost"):
        ContextExtraction(**bad)


def test_dangling_risk_related_node_is_rejected():
    bad = {**VALID, "risks": [{"id": "r1", "description": "x", "severity": "low", "related_nodes": ["ghost"]}]}
    with pytest.raises(ValidationError, match="ghost"):
        ContextExtraction(**bad)


def test_invalid_entity_type_is_rejected_by_schema():
    bad = {**VALID, "entities": [{"id": "a", "label": "A", "type": "not_a_real_type"}]}
    with pytest.raises(ValidationError):
        ContextExtraction(**bad)


def test_evidence_field_defaults_to_empty_string():
    extraction = ContextExtraction(**VALID)
    assert extraction.entities[0].evidence == ""
