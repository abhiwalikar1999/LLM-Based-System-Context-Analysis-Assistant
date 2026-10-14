# run_evals_model_comparison.py
#
# Phase 6: model comparison. Runs the SAME eval_cases.py cases through
# core.build_context_graph for two different models, reusing run_evals.py's
# exact scoring functions (no duplicated/drifted scoring logic) -- the
# default model (llm_engine.MODEL_NAME, gwdg.qwen3-30b-a3b-instruct-2507)
# and one alternative, chosen and verified before writing this script (see
# docs/PROGRESS.md): gwdg.meta-llama-3.1-8b-instruct -- a materially
# smaller model (8B vs the ~30B-class MoE default) on the SAME gateway, so
# the comparison isolates "which model" from "which provider/hosting",
# matching this project's general discipline of changing one variable at a
# time when possible.
#
# Why re-run the default model here instead of reusing Phase 4's saved
# baseline numbers: quality scores would still be valid reused, but LATENCY
# -- the other half of this phase's "quality vs. latency vs. cost" checklist
# item -- was never instrumented in the original run_evals.py runs, and
# comparing a freshly-timed alternative model against untimed historical
# data would be an unfair, apples-to-oranges comparison (network/gateway
# load varies run to run). Re-running both back to back, under the same
# conditions, in the same script, is the only way to make the latency
# comparison honest.
#
# Run from within neurocardiowatch_app3/, same convention as the other
# eval scripts:
#   python run_evals_model_comparison.py

import json
import os
import sys
import time
import traceback

from eval_cases import EVAL_CASES
from core import build_context_graph
from run_evals import (
    score_entities, score_relationships, score_text_coverage,
    score_evidence_grounding, score_relationship_label_normalization,
    macro_average,
)

MODELS = {
    "current": "gwdg.qwen3-30b-a3b-instruct-2507",
    "alternative": "gwdg.meta-llama-3.1-8b-instruct",
}


def run_one_case(case, model_name):
    start = time.time()
    raw_result, graph = build_context_graph(case["input_text"], model=model_name)
    elapsed = time.time() - start

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
        "elapsed_seconds": elapsed,
        "usage": raw_result.get("_usage"),
        "entity_scores": entity_scores,
        "relationship_scores": relationship_scores,
        "risk_coverage": risk_scores,
        "assumption_coverage": assumption_scores,
        "evidence_grounding": grounding_scores,
        "relationship_normalization": normalization_scores,
        "predicted_nodes": predicted_nodes,
        "predicted_edges": predicted_edges,
        "predicted_risks": predicted_risks,
        "predicted_assumptions": predicted_assumptions,
    }


def run_all_for_model(model_key, model_name):
    results, failures = [], []
    for case in EVAL_CASES:
        print(f"[{model_key}] Running case: {case['id']} ({case['domain']}) ...", file=sys.stderr)
        try:
            result = run_one_case(case, model_name)
            results.append(result)
            es, rs = result["entity_scores"], result["relationship_scores"]
            tokens = result["usage"]["total_tokens"] if result.get("usage") else "n/a"
            print(
                f"  entities F1={es['f1']:.2f} | relationships F1={rs['f1']:.2f} | "
                f"risk coverage={result['risk_coverage']['recall']:.2f} | "
                f"elapsed={result['elapsed_seconds']:.1f}s | tokens={tokens}",
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
            "avg_elapsed_seconds": macro_average([r["elapsed_seconds"] for r in results]),
            "avg_total_tokens": macro_average([r["usage"]["total_tokens"] for r in results if r.get("usage")]),
            "cases_run": len(results),
            "cases_failed": len(failures),
        }
    return aggregate, results, failures


def main():
    report = {}
    for model_key, model_name in MODELS.items():
        print(f"\n=== Model: {model_key} ({model_name}) ===", file=sys.stderr)
        aggregate, results, failures = run_all_for_model(model_key, model_name)
        report[model_key] = {"model_name": model_name, "aggregate": aggregate, "results": results, "failures": failures}
        print(f"--- Aggregate ({model_key}) ---", file=sys.stderr)
        for k, v in aggregate.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", file=sys.stderr)

    os.makedirs("eval_results", exist_ok=True)
    out_path = os.path.join("eval_results", f"model_comparison_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full report to {out_path}", file=sys.stderr)
    return report


if __name__ == "__main__":
    main()
