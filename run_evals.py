# run_evals.py
#
# Phase 1 evaluation harness. Run from within neurocardiowatch_app3/ (same
# convention as app_server.py) so the relative UPB_AI_GATEWAY_KEY.txt lookup
# in llm_engine.py resolves:
#
#   python run_evals.py
#
# What this measures, and why (see eval_cases.py's header comment for the
# full reasoning): the RAW LLM extraction stage -- core.build_context_graph,
# i.e. llm_engine.ask_chatgpt() plus canonicalization/dedup -- BEFORE the
# deterministic, hospital-specific enrichment layers run. That isolates
# exactly what varies with input text and prompt: the LLM's own reading
# comprehension, not the hand-tuned deterministic skeleton.
#
# Matching methodology (deliberately simple, deterministic, and documented
# rather than hidden -- a known simplification vs. embedding/LLM-judge
# matching, which would cost extra LLM calls and isn't needed to get a
# repeatable baseline number):
#   - Entities: normalized token-Jaccard similarity between labels, with a
#     crude trailing-"s" stem (so "nurses"/"nurse" still match), gated by an
#     exact type match (actor/component/external_system). Matched 1:1 via
#     greedy highest-score-first assignment (not a full optimal bipartite
#     match -- good enough at this scale: ~8-11 entities per case).
#   - Relationships: graded on whether the (source entity, target entity)
#     PAIR was found, reusing the entity alignment above. The edge's label
#     text (e.g. "streams_to" vs "forwards") is NOT graded here -- label
#     vocabulary consistency is explicitly Phase 3's job, not Phase 1's.
#   - Risks / assumptions: free-text, so graded as coverage (recall only):
#     for each hand-labeled risk/assumption, does any predicted one have
#     enough token overlap to count as "found"? We don't penalize the model
#     for listing additional, reasonable risks we didn't happen to write
#     down -- that's not the same kind of error as inventing a component.

import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict

from eval_cases import EVAL_CASES
from core import build_context_graph

ENTITY_MATCH_THRESHOLD = 0.34
TEXT_COVERAGE_THRESHOLD = 0.25

STOPWORDS = {"the", "a", "an", "of", "and", "to", "for", "in", "on", "via"}


def _stem(token: str) -> str:
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(label: str) -> set:
    # Strip hyphens before splitting so compound tech terms match regardless
    # of how they're written ("Wi-Fi" / "wifi" / "wi fi" all -> "wifi").
    # Found via a real miss during the first baseline run: see PROGRESS.md.
    normalized = (label or "").lower().replace("-", "")
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return {_stem(t) for t in tokens if t not in STOPWORDS}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def align_entities(predicted_nodes, gt_entities, threshold=ENTITY_MATCH_THRESHOLD):
    """
    predicted_nodes: list of dicts {id, label, type}
    gt_entities: list of dicts {label, type}
    Returns (pred_id_to_gt_index, matched_gt_indices_set).
    """
    candidates = []
    for p in predicted_nodes:
        p_tokens = tokenize(p["label"])
        for g_idx, g in enumerate(gt_entities):
            if p["type"] != g["type"]:
                continue
            score = jaccard(p_tokens, tokenize(g["label"]))
            if score >= threshold:
                candidates.append((score, p["id"], g_idx))
    candidates.sort(key=lambda x: -x[0])

    matched_pred, matched_gt = set(), set()
    pred_to_gt = {}
    for score, pid, gidx in candidates:
        if pid in matched_pred or gidx in matched_gt:
            continue
        matched_pred.add(pid)
        matched_gt.add(gidx)
        pred_to_gt[pid] = gidx
    return pred_to_gt, matched_gt


def score_entities(predicted_nodes, gt_entities):
    pred_to_gt, matched_gt = align_entities(predicted_nodes, gt_entities)
    precision = len(pred_to_gt) / len(predicted_nodes) if predicted_nodes else 0.0
    recall = len(matched_gt) / len(gt_entities) if gt_entities else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "matched": len(matched_gt), "predicted_count": len(predicted_nodes),
        "gt_count": len(gt_entities),
    }, pred_to_gt


def score_relationships(predicted_edges, gt_relationships, gt_entities, pred_to_gt):
    """
    pred_to_gt maps a predicted node id -> matched ground-truth entity INDEX
    (from align_entities). Resolve each ground-truth relationship's
    source/target labels to the same index space so the two can be compared.
    """
    gt_index_pairs = []
    for r in gt_relationships:
        src_idx = next((i for i, g in enumerate(gt_entities) if g["label"] == r["source"]), None)
        tgt_idx = next((i for i, g in enumerate(gt_entities) if g["label"] == r["target"]), None)
        gt_index_pairs.append((src_idx, tgt_idx))
    gt_index_pairs_set = set(gt_index_pairs)

    matched_gt = set()
    matched_pred_count = 0
    for e in predicted_edges:
        src_gidx = pred_to_gt.get(e["source"])
        tgt_gidx = pred_to_gt.get(e["target"])
        if src_gidx is None or tgt_gidx is None:
            continue
        pair = (src_gidx, tgt_gidx)
        if pair in gt_index_pairs_set:
            matched_pred_count += 1
            matched_gt.add(pair)

    precision = matched_pred_count / len(predicted_edges) if predicted_edges else 0.0
    recall = len(matched_gt) / len(gt_index_pairs) if gt_index_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "matched": len(matched_gt), "predicted_count": len(predicted_edges),
        "gt_count": len(gt_index_pairs),
    }


def score_text_coverage(predicted_texts, gt_texts, threshold=TEXT_COVERAGE_THRESHOLD):
    if not gt_texts:
        return {"recall": 1.0, "covered": 0, "gt_count": 0}
    pred_token_sets = [tokenize(t) for t in predicted_texts]
    covered = 0
    for gt in gt_texts:
        gt_tokens = tokenize(gt)
        best = max((jaccard(gt_tokens, pt) for pt in pred_token_sets), default=0.0)
        if best >= threshold:
            covered += 1
    return {"recall": covered / len(gt_texts), "covered": covered, "gt_count": len(gt_texts)}


def run_one_case(case):
    raw_result, graph = build_context_graph(case["input_text"])

    predicted_nodes = [{"id": n.id, "label": n.label, "type": n.type} for n in graph.nodes.values()]
    predicted_edges = [{"source": e.source, "target": e.target, "label": e.label} for e in graph.edges]
    predicted_risks = [r.description for r in graph.risks]
    predicted_assumptions = [a.description for a in graph.assumptions]

    gt = case["ground_truth"]

    entity_scores, pred_to_gt = score_entities(predicted_nodes, gt["entities"])
    relationship_scores = score_relationships(predicted_edges, gt["relationships"], gt["entities"], pred_to_gt)
    risk_scores = score_text_coverage(predicted_risks, gt["risks"])
    assumption_scores = score_text_coverage(predicted_assumptions, gt["assumptions"])

    return {
        "case_id": case["id"],
        "domain": case["domain"],
        "entity_scores": entity_scores,
        "relationship_scores": relationship_scores,
        "risk_coverage": risk_scores,
        "assumption_coverage": assumption_scores,
        "raw_llm_result": raw_result,
        "predicted_nodes": predicted_nodes,
        "predicted_edges": predicted_edges,
        "predicted_risks": predicted_risks,
        "predicted_assumptions": predicted_assumptions,
    }


def macro_average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    results = []
    failures = []

    for case in EVAL_CASES:
        print(f"Running case: {case['id']} ({case['domain']}) ...", file=sys.stderr)
        try:
            result = run_one_case(case)
            results.append(result)
            es, rs = result["entity_scores"], result["relationship_scores"]
            print(
                f"  entities P={es['precision']:.2f} R={es['recall']:.2f} F1={es['f1']:.2f} "
                f"({es['matched']}/{es['gt_count']} matched, {es['predicted_count']} predicted) | "
                f"relationships P={rs['precision']:.2f} R={rs['recall']:.2f} F1={rs['f1']:.2f} | "
                f"risk coverage={result['risk_coverage']['recall']:.2f} "
                f"assumption coverage={result['assumption_coverage']['recall']:.2f}",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            traceback.print_exc()
            failures.append({"case_id": case["id"], "error": str(e)})
        time.sleep(1)  # stay well under the gwdg backend's rate limit

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
            "cases_run": len(results),
            "cases_failed": len(failures),
        }

    print("\n=== Aggregate (macro-average across cases) ===", file=sys.stderr)
    for k, v in aggregate.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}", file=sys.stderr)

    os.makedirs("eval_results", exist_ok=True)
    out_path = os.path.join("eval_results", f"baseline_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "results": results, "failures": failures}, f, indent=2)
    print(f"\nSaved full report to {out_path}", file=sys.stderr)

    return aggregate, results, failures


if __name__ == "__main__":
    main()
