# agentic.py
#
# Phase 5: agentic self-check / validation loop, implemented as a small
# LangGraph graph: extract -> critique -> (conditional) -> repair -> back to
# critique; clean -> end.
#
# Why LangGraph here and nowhere else in this codebase: this is the one
# place in the whole pipeline with a genuine conditional CYCLE (critique
# finds an issue -> repair -> re-check; clean -> stop). Phase 2's retry-on-
# invalid-schema loop, by contrast, is a plain Python `for` loop in
# llm_engine.py -- a single bounded retry doesn't need a graph framework,
# and reaching for one there would be unjustified complexity. Reserving
# LangGraph for the one phase that structurally earns it (see
# docs/ROADMAP.md, Phase 5) keeps the choice defensible rather than
# resume-driven.
#
# Scope: this wraps core.build_context_graph (Phases 0-4's existing
# extraction pipeline, including retrieval) as the "extract" node, and adds
# two NEW LLM calls on top -- critique and repair -- both using the same
# structured-output discipline as Phase 2 (Pydantic schema, not free text).
# Repair operates against the FULL concatenated source text, not just
# retrieval's selected subset: repair only runs when something's flagged
# (relatively rare), so it can afford to look at everything, and doing so
# gives it the best chance of actually finding what retrieval may have
# filtered out of the initial extraction pass.

import json
from typing import List, Literal, Tuple, TypedDict, Union

from langgraph.graph import StateGraph, END
from openai import OpenAI
from pydantic import BaseModel

from llm_engine import _get_key, BASE_URL, MODEL_NAME, ContextExtraction
from core import build_context_graph, canonicalize_extraction, ContextGraph

MAX_ITERATIONS = 3  # per docs/ROADMAP.md: "2-3 rounds" cap, so a stubborn
                     # disagreement between critique and repair can't loop
                     # indefinitely or run up cost unbounded.


# --- Critique schema -----------------------------------------------------

class CritiqueIssue(BaseModel):
    issue_type: Literal["missing_actor", "invented_relationship", "contradiction", "other"]
    description: str


class CritiqueResult(BaseModel):
    issues: List[CritiqueIssue] = []


CRITIQUE_SYSTEM_PROMPT = """
You are a careful reviewer checking an automated system-context extraction against its source text.

Given the SOURCE TEXT and the EXTRACTED JSON (entities, relationships, risks, assumptions), identify concrete
issues in exactly these categories:
- missing_actor: a human role or actor explicitly described as acting in the text (e.g. "Riders request...",
  "Support agents use...") that is NOT present as an entity with type "actor" in the extraction.
- invented_relationship: a relationship in the extraction whose (source, target) connection is not actually
  stated or clearly implied anywhere in the source text.
- contradiction: an entity or relationship that contradicts something else explicitly stated in the text
  (e.g. an entity classified as the wrong type given how the text describes it).

Only report issues you can point to concretely in the text. If there are no issues, return an empty issues
list. Do not report stylistic preferences, missing optional detail, or things that are merely unmentioned --
only an actual, checkable discrepancy between the text and the extraction.
"""


def critique_extraction(source_text: str, extraction: dict) -> CritiqueResult:
    client = OpenAI(api_key=_get_key(), base_url=BASE_URL)
    user_content = f"SOURCE TEXT:\n{source_text}\n\nEXTRACTED JSON:\n{json.dumps(extraction, indent=2)}"
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=CritiqueResult,
        temperature=0,
    )
    message = response.choices[0].message
    if message.refusal:
        return CritiqueResult(issues=[])
    return message.parsed


# --- Repair ----------------------------------------------------------------

REPAIR_SYSTEM_PROMPT = """
You previously extracted system-context information from the SOURCE TEXT below, producing the PRIOR
EXTRACTION. A reviewer found the ISSUES listed below. Produce a corrected, complete extraction in the same
JSON schema, fixing exactly these issues while preserving everything else that was already correct. Do not
remove or change anything the reviewer did not flag.
"""


def repair_extraction(source_text: str, prior_extraction: dict, issues: List[CritiqueIssue]) -> dict:
    client = OpenAI(api_key=_get_key(), base_url=BASE_URL)
    issues_text = "\n".join(f"- [{i.issue_type}] {i.description}" for i in issues)
    user_content = (
        f"SOURCE TEXT:\n{source_text}\n\n"
        f"PRIOR EXTRACTION:\n{json.dumps(prior_extraction, indent=2)}\n\n"
        f"ISSUES TO FIX:\n{issues_text}"
    )
    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format=ContextExtraction,
        temperature=0,
    )
    message = response.choices[0].message
    if message.refusal:
        return prior_extraction  # fall back to the prior extraction on refusal
    return message.parsed.model_dump()


# --- LangGraph wiring --------------------------------------------------

class AgenticState(TypedDict, total=False):
    documents: Union[str, List[Tuple[str, str]]]
    source_text: str          # full concatenated source text, for critique/repair
    extraction: dict
    graph: ContextGraph
    issues: List[CritiqueIssue]
    iteration: int
    history: List[dict]       # one entry per critique pass, for transparency/eval


def _node_extract(state: AgenticState) -> dict:
    result, graph = build_context_graph(state["documents"])
    return {"extraction": result, "graph": graph, "iteration": 0, "history": []}


def _node_critique(state: AgenticState) -> dict:
    critique = critique_extraction(state["source_text"], state["extraction"])
    entry = {"iteration": state["iteration"], "issues_found": len(critique.issues),
              "issue_types": [i.issue_type for i in critique.issues]}
    return {"issues": critique.issues, "history": state["history"] + [entry]}


def _node_repair(state: AgenticState) -> dict:
    repaired = repair_extraction(state["source_text"], state["extraction"], state["issues"])
    graph = canonicalize_extraction(repaired)
    return {"extraction": repaired, "graph": graph, "iteration": state["iteration"] + 1}


def _route_after_critique(state: AgenticState) -> str:
    if state["issues"] and state["iteration"] < MAX_ITERATIONS:
        return "repair"
    return "end"


def _build_graph():
    g = StateGraph(AgenticState)
    g.add_node("extract", _node_extract)
    g.add_node("critique", _node_critique)
    g.add_node("repair", _node_repair)
    g.set_entry_point("extract")
    g.add_edge("extract", "critique")
    g.add_conditional_edges("critique", _route_after_critique, {"repair": "repair", "end": END})
    g.add_edge("repair", "critique")
    return g.compile()


_COMPILED_GRAPH = _build_graph()


def _concat_source_text(documents: Union[str, List[Tuple[str, str]]]) -> str:
    if isinstance(documents, str):
        return documents
    return "\n\n".join(text for _, text in documents)


def run_agentic_pipeline(documents: Union[str, List[Tuple[str, str]]]) -> Tuple[dict, ContextGraph, dict]:
    """
    Runs the full extract -> critique -> (repair -> critique)* loop.
    Returns (final_extraction_dict, final_ContextGraph, run_metadata), where
    run_metadata includes the per-iteration critique history and total LLM
    calls made (1 extract call + 1 critique per iteration + 1 repair per
    repair round) -- the cost side of the "quality delta vs cost delta"
    tradeoff this phase is meant to measure.
    """
    initial_state: AgenticState = {
        "documents": documents,
        "source_text": _concat_source_text(documents),
    }
    final_state = _COMPILED_GRAPH.invoke(initial_state)

    history = final_state["history"]
    repair_rounds = final_state["iteration"]
    total_llm_calls = 1 + len(history) + repair_rounds  # extract + critiques + repairs

    metadata = {
        "history": history,
        "repair_rounds": repair_rounds,
        "total_llm_calls": total_llm_calls,
        "final_issue_count": len(final_state.get("issues") or []),
    }
    return final_state["extraction"], final_state["graph"], metadata
