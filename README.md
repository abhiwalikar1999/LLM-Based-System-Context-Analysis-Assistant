# LLM-Based System Context Analysis Assistant

[![CI](https://github.com/abhiwalikar1999/LLM-Based-System-Context-Analysis-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/abhiwalikar1999/LLM-Based-System-Context-Analysis-Assistant/actions/workflows/ci.yml)

**Live demo:** [context-graph-assistant.onrender.com](https://context-graph-assistant.onrender.com) — see the **Deployment** section below for why the live "Analyze" button won't produce new results (the real, pre-generated example output on that page will).

An AI-assisted tool that turns a plain-English system description into an interactive **system-context model**: the actors, components, and external systems involved, how they relate, and what risks/assumptions are implied — extracted automatically by an LLM instead of drawn by hand.

## The problem

Early in designing any non-trivial system, you need a shared picture of its context: who's involved, what talks to what, where the system boundary is, what could go wrong. Normally someone draws this by hand from a requirements document, and it's stale by the next meeting. This project explores using an LLM to extract that context model directly from text — and keep it current as the description evolves.

## Approach

The organizing principle behind every change here: measure before changing anything, then measure again after. A 6-case hand-labeled evaluation harness was built *before* any pipeline change, not retrofitted afterward to look rigorous — every later addition (structured outputs, evidence traceability, retrieval, an agentic self-check loop, a head-to-head model comparison) was re-run against that same harness, and the numbers below are reported as measured, including the ones that weren't flattering. A smaller, faster-looking model turned out to produce zero usable output on the real task — a finding only the eval harness caught, since a quick manual test of the same model had looked fine. An agentic self-check loop that measurably improved entity recall also cost ~4.7× more per analysis, which is exactly why it isn't the default path. The goal throughout was a defensible before/after story, not a flattering one.

## Results at a glance

| | |
|---|---|
| **Baseline extraction quality** (6 hand-labeled cases, varied domains) | Entity F1 0.88 · Relationship F1 0.82 · Evidence grounding 1.00 |
| **Agentic self-check loop** vs. the plain pipeline | Entity F1 **+0.045**, traced to real fixes (not luck) — at **~4.7×** the LLM calls, so not shipped as the default |
| **Model comparison**: default vs. a smaller alternative | Alternative was 4.6× faster at equal token cost, but scored **0.00** on the core task — caught by the eval harness, missed by a quick manual test |
| **Software quality** | 65 unit tests, 0 network calls, CI on every push (tests + a real Docker build + container smoke-test) |

Full methodology, raw numbers, and the investigation behind each finding are in the sections below.

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

A separate, opt-in **agentic validation loop** (`agentic.py`, LangGraph) wraps steps 1–2 with a self-check: `extract → critique → repair → critique...` until clean or a 3-round cap. Not wired in as the default path — see **Agentic validation loop** below for why.

## Tech stack

Python, a plain `http.server`-based backend (no web framework, by design), an OpenAI-compatible LLM API client, and `pyvis`/`vis-network` for the graph visualizations.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # only needed to run the test suite
```

Provide an OpenAI-compatible API key via the `UPB_AI_GATEWAY_KEY` environment variable, or a local `UPB_AI_GATEWAY_KEY.txt` file (gitignored — never commit this). The gateway URL/model are set in `llm_engine.py`; point them at whatever OpenAI-compatible endpoint you have access to.

```bash
python app_server.py
```

Open `http://127.0.0.1:8000`, paste text and/or upload one or more `.txt` files (see `example_system_description.txt` for a sample input describing a hospital patient-monitoring system) — multiple files are treated as separate documents and aggregated into one context model — and click **Analyze**.

**Or with Docker:**

```bash
docker build -t context-graph-assistant .
docker run -p 8000:8000 -e UPB_AI_GATEWAY_KEY=your-key-here context-graph-assistant
```

## Tests

```bash
python -m pytest
```

65 unit tests across 5 files, ~1.5s, zero network calls (everything exercised via monkeypatching rather than hitting the real gateway). Covers the deterministic graph logic (deduplication, canonicalization, relationship normalization), the eval harness's own scoring functions, the structured-output schema's validation, and the agentic loop's routing/iteration-cap logic. Largely a formalization of checks this project was already running ad hoc at every phase — see `tests/` for the per-module breakdown.

## Deployment

The app is containerized (`Dockerfile`) with CI (`.github/workflows/ci.yml`) that runs the test suite on every push, then — since there's no Docker available in this project's dev environment — builds the real image on GitHub's runners and smoke-tests the running container (home page serves, bundled example output is reachable from inside the container). That CI run *is* the verification that the Dockerfile actually works, not an assumption.

**A real constraint worth being upfront about:** the LLM backend (UPB AI Gateway) is only reachable from the Paderborn University network/VPN. A publicly hosted instance of this app genuinely cannot reach it — clicking "Analyze" on a public deployment will time out, which is an access restriction on the backend, not a bug in this code. Rather than hide that or ship something that looks broken, the app bundles a real, pre-generated `example_output/` (committed to the repo, not regenerated at request time) that's always viewable without a live call — see the home page's "Example output" section.

Deployed to [Render](https://render.com) via `render.yaml` (a Blueprint) — **[context-graph-assistant.onrender.com](https://context-graph-assistant.onrender.com)**. Render builds and deploys the Dockerfile directly from this repo on every push; the one secret (`UPB_AI_GATEWAY_KEY`) is set in Render's dashboard, never read from the repo. `app_server.py`'s `HOST`/`PORT` read from the environment specifically for this — `0.0.0.0` and Render's injected port, rather than the `127.0.0.1` default used for local runs.

Worth being precise about one thing: the VPN requirement is about *the server*, not the visitor. Now that the backend calls are made by Render's infrastructure instead of a local machine, no amount of the visitor being on the UPB VPN changes anything — Render's servers aren't on that network and can't be. That's exactly why the bundled example output exists: it's real output, generated once from a VPN-connected run and committed to the repo, so the live page always has something genuine to show.

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
- An agentic self-check/repair loop (LangGraph) that reviews its own extraction against the source text and corrects specific, named issues — available as a standalone capability (`agentic.py`), measured, and deliberately not wired in as the default path (see below for the cost/quality tradeoff that decision was based on)
- A model-comparison harness (`run_evals_model_comparison.py`) that swaps the extraction model and re-runs the full eval suite, measuring quality, latency, and token cost side by side — see below for a finding it caught that a quick manual test completely missed
- A unit test suite (`tests/`, pytest) for the pure-Python pipeline logic — deduplication, canonicalization, the eval harness's own scoring functions, schema validation, and the agentic loop's routing logic — 65 tests, zero network calls, ~1.5s
- Dockerized, with CI that builds the real image and smoke-tests the running container on every push (not just "it compiles") — see **Deployment** below

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

## Agentic validation loop

`agentic.py` implements a second, opt-in pipeline using [LangGraph](https://github.com/langchain-ai/langgraph): `extract → critique → repair → critique...` until the critique pass finds nothing wrong, or a 3-round cap is hit. Critique checks for 3 concrete, checkable issue types — missing actors, invented relationships, contradictions — using the same structured-output approach as the main pipeline, not free-text review.

Ran it against the same 6 eval cases (`run_evals_agentic.py`) and compared against the plain pipeline:

| Metric | Plain | Agentic | Delta |
|---|---|---|---|
| Entity F1 | 0.879 | 0.924 | **+0.045** |
| Relationship F1 | 0.820 | 0.824 | +0.004 (flat) |
| Avg. LLM calls per analysis | 1.0 | 4.7× | — |

The entity-recall gain is real and traceable, not a lucky score movement: in every case that triggered a repair, the fix was a previously-missing actor entity getting added back — the exact failure mode this project has documented since its very first connectivity test. The cost is real too: ~4.7× more LLM calls on average, with a worst case of 8 calls and ~5.5 minutes for one multi-document analysis. One case converged cleanly after 3 rounds; another plateaued at the same 3 unresolved issues for all 3 rounds without ever reaching clean — concrete evidence the iteration cap is a necessary safeguard, not just caution.

Given that cost profile, it's **not** wired in as the app's default path — that would need a considered redesign of the session request-budget accounting this project's guardrails weren't built around. It's available and fully tested as its own capability.

Run it yourself:

```bash
python run_evals_agentic.py
```

## Model comparison

`run_evals_model_comparison.py` swaps the extraction model and re-runs the full eval suite — both models fresh, back to back, under the same conditions — to compare quality, latency, and token cost together rather than picking a model on vibes.

Compared the default model against a materially smaller one on the same gateway (isolating "which model" from "which provider"):

| Metric | Default (~30B-class) | Smaller (8B) |
|---|---|---|
| Entity F1 | 0.863 | **0.000** |
| Relationship F1 | 0.809 | **0.000** |
| Avg. latency | 42.4s | 9.2s (4.6× faster) |
| Avg. tokens | 1831 | 1861 (about equal) |

On paper the smaller model looks like a clean win — 4.6× faster at roughly equal token cost. It isn't: it scored 0.00 on the core extraction task in **every single one of 6 test cases**. Didn't take that number at face value — inspecting the raw output showed the model's `label` field was literally the string `"component"`/`"actor"` (the type name, not an entity name) every time, which the pipeline's own deduplication collapses into one placeholder node per type instead of the real 8–15 entities expected. Free-text fields (risks, assumptions) stayed accurate throughout, isolating the failure to the structured entity/relationship schema specifically — and notably, this same model had handled a single trivial test sentence correctly earlier, so the failure is tied to task complexity, not a blanket incapability. A quick manual spot-check would have missed this entirely; only the eval harness caught it. The lesson: a faster, cheaper-looking model is worthless if it fails the actual task, and knowing that requires an eval harness, not a vibe check.

Run it yourself:

```bash
python run_evals_model_comparison.py
```
