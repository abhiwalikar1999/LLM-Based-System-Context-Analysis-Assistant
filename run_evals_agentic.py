# run_evals_agentic.py
#
# Phase 5 comparison harness. Runs the SAME eval_cases.py cases and reuses
# run_evals.py's exact scoring functions (no duplicated/drifted scoring
# logic), but through agentic.run_agentic_pipeline instead of
# core.build_context_graph -- measuring the agentic critique/repair loop's
# quality delta AND its cost delta (extra LLM calls/latency) against the
# Phase 4 baseline, per docs/ROADMAP.md's Phase 5 checklist.
#
# Run from within neurocardiowatch_app3/, same convention as run_evals.py:
#   python run_evals_agentic.py

import json
import os
import sys
import time
import traceback

from eval_cases import EVAL_CASES
from agentic import run_agentic_pipeline
from run_evals import (
    score_entities, score_relationships, score_text_coverage,
    score_evidence_grounding, score_relationship_label_normalization,
    macro_average,
)


def run_one_case_agentic(case):
    extraction, graph, agentic_meta = run_agentic_pipeline(case["input_text"])

    predicted_nodes = [{"id": n.id, "label": n.label, "type": n.type} for n in graph.nodes.values()]
    predicted_edges = [{"source": e.source, "target": e.target, "label": e.label} for e in graph.edges]
    predicted_risks = [r.description for r in graph.risks]
    predicted_assumptions = [a.description for a in graph.assumptions]

    all_evidence = (
        [n.evidence for n in graph.nodes.values()]
        + [e.evidence for e in graph.edges]
        + [r.evidence for r in graph.risks]
        + [a.evidence for a in graph.assumptions]
    )

    if isinstance(case["input_text"], str):
        source_text_for_grounding = case["input_text"]
    else:
        source_text_for_grounding = "\n\n".join(text for _, text in case["input_text"])

    gt = case["ground_truth"]

    entity_scores, pred_to_gt = score_entities(predicted_nodes, gt["entities"])
    relationship_scores = score_relationships(predicted_edges, gt["relationships"], gt["entities"], pred_to_gt)
    risk_scores = score_text_coverage(predicted_risks, gt["risks"])
    assumption_scores = score_text_coverage(predicted_assumptions, gt["assumptions"])
    grounding_scores = score_evidence_grounding(all_evidence, source_text_for_grounding)
    normalization_scores = score_relationship_label_normalization(predicted_edges)

    return {
        "case_id": case["id"],
        "domain": case["domain"],
        "entity_scores": entity_scores,
        "relationship_scores": relationship_scores,
        "risk_coverage": risk_scores,
        "assumption_coverage": assumption_scores,
        "evidence_grounding": grounding_scores,
        "relationship_normalization": normalization_scores,
        "agentic_meta": agentic_meta,
        "predicted_nodes": predicted_nodes,
        "predicted_edges": predicted_edges,
        "predicted_risks": predicted_risks,
        "predicted_assumptions": predicted_assumptions,
    }


def main():
    results = []
    failures = []

    for case in EVAL_CASES:
        print(f"Running case (agentic): {case['id']} ({case['domain']}) ...", file=sys.stderr)
        start = time.time()
        try:
            result = run_one_case_agentic(case)
            results.append(result)
            es, rs = result["entity_scores"], result["relationship_scores"]
            am = result["agentic_meta"]
            print(
                f"  entities P={es['precision']:.2f} R={es['recall']:.2f} F1={es['f1']:.2f} | "
                f"relationships P={rs['precision']:.2f} R={rs['recall']:.2f} F1={rs['f1']:.2f} | "
                f"risk coverage={result['risk_coverage']['recall']:.2f} "
                f"assumption coverage={result['assumption_coverage']['recall']:.2f} | "
                f"repair_rounds={am['repair_rounds']} total_llm_calls={am['total_llm_calls']} "
                f"elapsed={time.time()-start:.1f}s",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            failures.append({"case_id": case["id"], "error": str(e)})
        time.sleep(1)

    aggregate = {}
    if results:
        aggregate = {
            "entity_precision": macro_average([r["entity_scores"]["precision"] for r in results]),
            "entity_recall": macro_average([r["entity_scores"]["recall"] for r in results]),
            "entity_f1": macro_average([r["entity_scores"]["f1"] for r in results]),
            "relationship_precision": macro_average([r["relationship_scores"]["precision"] for r in results]),
            "relationship_recall": macro_average([r["relationship_scores"]["recall"] for r in results]),
            "relationship_f1": macro_average([r["relationship_scores"]["f1"] for r in results]),
            "risk_coverage": macro_average([r["risk_coverage"]["recall"] for r in results]),
            "assumption_coverage": macro_average([r["assumption_coverage"]["recall"] for r in results]),
            "evidence_grounded_rate": macro_average([r["evidence_grounding"]["grounded_rate"] for r in results]),
            "relationship_label_normalized_rate": macro_average([r["relationship_normalization"]["normalized_rate"] for r in results]),
            "avg_repair_rounds": macro_average([r["agentic_meta"]["repair_rounds"] for r in results]),
            "avg_total_llm_calls": macro_average([r["agentic_meta"]["total_llm_calls"] for r in results]),
            "cases_run": len(results),
            "cases_failed": len(failures),
        }

    print("\n=== Aggregate (macro-average across cases, agentic pipeline) ===", file=sys.stderr)
    for k, v in aggregate.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", file=sys.stderr)

    os.makedirs("eval_results", exist_ok=True)
    out_path = os.path.join("eval_results", f"agentic_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "results": results, "failures": failures}, f, indent=2)
    print(f"\nSaved full report to {out_path}", file=sys.stderr)

    return aggregate, results, failures


if __name__ == "__main__":
    main()
