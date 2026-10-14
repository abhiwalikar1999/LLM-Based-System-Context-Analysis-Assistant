# tests/test_agentic.py
#
# Phase 7: formalizes the Phase 5 graph-control-flow checks (clean pass,
# one repair, capped-at-max-iterations) into a repeatable test suite. No
# network: extract/critique/repair are all monkeypatched.

import agentic
from agentic import CritiqueIssue, CritiqueResult
from core import ContextGraph

FAKE_EXTRACTION = {"entities": [], "relationships": [], "risks": [], "assumptions": []}


def _fake_extract(documents):
    return dict(FAKE_EXTRACTION), ContextGraph()


def test_clean_on_first_pass_requires_no_repair(monkeypatch):
    monkeypatch.setattr(agentic, "build_context_graph", _fake_extract)
    monkeypatch.setattr(agentic, "critique_extraction", lambda src, ext: CritiqueResult(issues=[]))

    extraction, graph, meta = agentic.run_agentic_pipeline("some text")
    assert meta["repair_rounds"] == 0
    assert meta["total_llm_calls"] == 2  # 1 extract + 1 critique
    assert meta["final_issue_count"] == 0


def test_one_issue_then_clean_after_repair(monkeypatch):
    monkeypatch.setattr(agentic, "build_context_graph", _fake_extract)

    call_count = {"n": 0}

    def fake_critique(source_text, extraction):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return CritiqueResult(issues=[CritiqueIssue(issue_type="missing_actor", description="missing rider")])
        return CritiqueResult(issues=[])

    monkeypatch.setattr(agentic, "critique_extraction", fake_critique)
    monkeypatch.setattr(agentic, "repair_extraction", lambda src, prior, issues: dict(FAKE_EXTRACTION))

    extraction, graph, meta = agentic.run_agentic_pipeline("some text")
    assert meta["repair_rounds"] == 1
    assert meta["total_llm_calls"] == 4  # 1 extract + 2 critiques + 1 repair
    assert meta["final_issue_count"] == 0


def test_persistent_disagreement_is_capped_at_max_iterations(monkeypatch):
    monkeypatch.setattr(agentic, "build_context_graph", _fake_extract)
    monkeypatch.setattr(
        agentic, "critique_extraction",
        lambda src, ext: CritiqueResult(issues=[CritiqueIssue(issue_type="contradiction", description="still wrong")]),
    )
    monkeypatch.setattr(agentic, "repair_extraction", lambda src, prior, issues: dict(FAKE_EXTRACTION))

    extraction, graph, meta = agentic.run_agentic_pipeline("some text")
    assert meta["repair_rounds"] == agentic.MAX_ITERATIONS  # stopped, not infinite
    assert meta["final_issue_count"] == 1  # still unresolved when the cap was hit
