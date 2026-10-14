# LLM-Based System Context Analysis Assistant

An AI-assisted tool that turns a plain-English system description into an interactive **system-context model**: the actors, components, and external systems involved, how they relate, and what risks/assumptions are implied — extracted automatically by an LLM instead of drawn by hand.

## The problem

Early in designing any non-trivial system, you need a shared picture of its context: who's involved, what talks to what, where the system boundary is, what could go wrong. Normally someone draws this by hand from a requirements document, and it's stale by the next meeting. This project explores using an LLM to extract that context model directly from text — and keep it current as the description evolves.

## How it works

```mermaid
flowchart LR
    A[System description(s)<br/>one or more documents] --> R[Retrieval<br/>chunk + embed +<br/>select relevant chunks]
    R --> B[LLM extraction<br/>entities, relationships,<br/>risks, assumptions]
    B --> C[Canonicalization<br/>dedupe + normalize]
    C --> D[Deterministic enrichment<br/>requirements, controls,<br/>governance layer]
    D --> E1[Context graph<br/>HTML]
    D --> E2[Risk overlay<br/>HTML]
    D --> E3[Governance overlay<br/>HTML]
    D --> F[Structured JSON<br/>model]
```

1. **Retrieval** — one or more input documents are chunked, embedded, and the chunks most relevant to each of 6 fixed extraction categories (actors, components, external systems, relationships, risks, assumptions) are selected via cosine similarity, instead of dumping the whole corpus into one prompt. Below a size threshold this is a no-op passthrough (small single-document input, unaffected).
2. **Extraction** — the selected text is sent to an LLM with a strict prompt, using schema-constrained structured output (not free-text JSON parsing) to guarantee a valid shape: entities (actors / internal components / external systems), relationships, risks, and assumptions, each with the supporting source sentence(s) attached. A referential-integrity check (every id referenced actually exists) catches what schema validation alone can't, with an automatic retry/repair pass on failure.
3. **Canonicalization** — extracted entities are deduplicated by normalized label + type (content-addressed IDs), so near-duplicate names don't create orphan nodes. Relationship labels (e.g. `streams_to`/`sends_signals_to`/`provides_alerts_to`) are normalized against a small canonical verb set, so the diagram doesn't accumulate inconsistent phrasing for the same kind of connection.
4. **Deterministic enrichment** — a rule-based layer fills in requirements (functional/non-functional), security & governance controls, and operational concerns, and links risks to the controls that mitigate them. This keeps the output complete and useful even when the LLM's raw extraction is sparse.
5. **Visualization** — three separate interactive graph views are rendered (via [pyvis](https://pyvis.readthedocs.io/)) instead of one overloaded diagram: a clean architecture/context view, a risk-traceability overlay, and a governance/requirements overlay.
6. **Caching** — outputs are cached by a hash of (input documents + pipeline version), so repeated identical input doesn't re-trigger an LLM call.

## Tech stack

Python, a plain `http.server`-based backend (no web framework, by design), an OpenAI-compatible LLM API client, and `pyvis`/`vis-network` for the graph visualizations.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
```

Provide an OpenAI-compatible API key via the `UPB_AI_GATEWAY_KEY` environment variable, or a local `UPB_AI_GATEWAY_KEY.txt` file (gitignored — never commit this). The gateway URL/model are set in `llm_engine.py`; point them at whatever OpenAI-compatible endpoint you have access to.

```bash
python app_server.py
```

Open `http://127.0.0.1:8000`, paste text and/or upload one or more `.txt` files (see `example_system_description.txt` for a sample input describing a hospital patient-monitoring system) — multiple files are treated as separate documents and aggregated into one context model — and click **Analyze**.

## What's built

- Working end-to-end pipeline: extraction → graph → 3 interactive views + JSON export
- Deterministic entity de-duplication and canonicalization via content-hashed IDs, so near-duplicate labels don't create orphan nodes
- Automatic requirements, controls, and governance enrichment layered onto the extracted model
- 3 separate interactive graph views per run (context, risk, governance) plus 1 structured JSON export, instead of one overloaded diagram
- Input-hash-based caching — re-analyzing identical input costs zero additional LLM calls
- A per-session request cap as a built-in cost-control guardrail
- An evaluation harness (`run_evals.py`) with 5 hand-labeled test cases across varied domains (healthcare, e-commerce, IoT, fintech, logistics) — not just the one reference scenario — measuring entity/relationship extraction quality and risk/assumption coverage against ground truth
- Schema-constrained structured LLM output (Pydantic + strict JSON schema) with a referential-integrity check and an automatic retry/repair loop, replacing free-text JSON parsing
- Evidence traceability — every extracted entity/relationship/risk/assumption carries the source sentence(s) it came from, visible as hover tooltips in the diagrams, not just buried in the JSON export
- Relationship-label normalization against a canonical verb set, built from labels actually observed in practice rather than guessed
- Multi-document input with retrieval — upload several files at once; each is chunked, embedded, and the most relevant chunks per extraction category are selected via cosine similarity rather than concatenating everything into one prompt

## Evaluation

Extraction quality is measured, not assumed, and changes are measured again after — not just assumed to have helped. `eval_cases.py` holds 6 hand-labeled test cases across varied domains, including one multi-document case; `run_evals.py` runs the extraction pipeline against each and scores it with fuzzy entity matching (type + token-overlap), relationship-pair matching, risk/assumption coverage, evidence groundedness, and relationship-label normalization coverage.

Current baseline (macro-average across all 6 cases):

| Metric | Score |
|---|---|
| Entity precision / recall / F1 | 0.96 / 0.82 / 0.88 |
| Relationship precision / recall / F1 | 0.81 / 0.84 / 0.82 |
| Risk coverage | 0.94 |
| Assumption coverage | 1.00 |
| Evidence grounded rate | 1.00 |
| Relationship label normalized rate | 0.47 |

These numbers mix two different things on purpose: 5 single-document cases that should be unaffected by the latest change (retrieval doesn't activate below a word-count threshold), and 1 new multi-document case testing retrieval directly. Checked that separation explicitly rather than reading one blended number — the 5 original cases moved only within already-established run-to-run noise. The multi-document case's own numbers looked weaker in isolation (entity F1 0.79, risk coverage 0.67) until investigated: the "missed" risk was actually extracted almost verbatim but scored just under the matching threshold due to paraphrasing (true coverage: 3/3), and several "unmatched" entities turned out to be gaps in the hand-labeled ground truth, not model errors (checked against the source text, not assumed). The real, separately-verified finding: multi-document aggregation works correctly end-to-end, and the retrieval filtering logic itself is confirmed correct (verified with a controlled test), but this particular test corpus was too small to show retrieval actually discarding irrelevant content in practice — the selection budget exceeded the chunk count, so nothing got filtered out this time.

Run it yourself:

```bash
python run_evals.py
```

Full methodology and per-case findings are documented in the script's header comments and `eval_cases.py`.
