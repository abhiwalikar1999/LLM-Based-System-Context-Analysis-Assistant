# retrieval.py
#
# Phase 4: retrieval / multi-document aggregation. Turns a list of
# (document_label, document_text) pairs into ONE selected text blob fed to
# the existing single-call extraction (core.build_context_graph ->
# llm_engine.ask_chatgpt) -- this is deliberately a pre-processing/input-
# selection step, not a new extraction mechanism, keeping the "second LLM
# pass" / multi-step orchestration sophistication reserved for Phase 5
# (LangGraph), where it's architecturally justified, rather than reused
# here where it isn't.
#
# Why retrieval and not just concatenation: for a handful of short documents
# that together still fit comfortably in the model's context, concatenation
# is simpler and loses nothing -- so retrieval only activates once the
# combined corpus exceeds SIZE_THRESHOLD_WORDS (see retrieve_relevant_chunks).
# Below that, this module is a no-op passthrough, which also means Phases
# 1-3's existing single-document eval cases are completely unaffected by
# this change -- verified, not just assumed (see docs/PROGRESS.md).
#
# No vector-DB / numpy dependency: at the scale this app operates at (a
# handful of documents, tens of chunks), a pure-Python cosine similarity
# over a plain list of embeddings is simpler, dependency-free, and fast
# enough -- consistent with the project's existing minimal-dependency style
# (see CLAUDE.md's note on app_server.py being framework-free on purpose).

import math
import re
from typing import List, Tuple

from openai import OpenAI
from llm_engine import _get_key, BASE_URL

EMBEDDING_MODEL = "gwdg.qwen3-embedding-4b"

SIZE_THRESHOLD_WORDS = 350  # below this, retrieval is a no-op passthrough.
# Calibrated against real data, not guessed: Phases 1-3's 5 existing
# single-document eval cases are all 139-166 words (checked directly before
# picking this number), so 350 comfortably keeps every one of them on the
# no-op passthrough path -- their established baselines stay untouched by
# Phase 4 -- while still being low enough that a genuinely multi-document
# input (Phase 4's own new eval case) clears it and exercises real retrieval.
CHUNK_MAX_WORDS = 150       # paragraphs longer than this get sentence-split
TOP_K_PER_QUERY = 3

# Fixed category queries representing what the extraction prompt is looking
# for -- embedded once, then used to retrieve the most relevant chunks for
# each category across ALL input documents (not "answer a question", but
# "surface the passages a human analyst would read for this category").
RETRIEVAL_QUERIES = [
    "actors and users who interact with the system, such as patients, customers, operators, or administrators",
    "internal components, services, and devices that make up the system",
    "external systems, third-party services, and integrations the system depends on",
    "data flows, interactions, and relationships between components",
    "risks, failure modes, and things that could go wrong",
    "assumptions and preconditions the system relies on",
]


def _word_count(text: str) -> int:
    return len((text or "").split())


def chunk_document(text: str, max_words: int = CHUNK_MAX_WORDS) -> List[str]:
    """Paragraph-based chunking (split on blank lines); long paragraphs are
    further split by sentence so no single chunk dominates retrieval."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    chunks = []
    for p in paragraphs:
        if _word_count(p) <= max_words:
            chunks.append(p)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = []
        buf_words = 0
        for s in sentences:
            w = _word_count(s)
            if buf and buf_words + w > max_words:
                chunks.append(" ".join(buf))
                buf, buf_words = [], 0
            buf.append(s)
            buf_words += w
        if buf:
            chunks.append(" ".join(buf))
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=_get_key(), base_url=BASE_URL)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_relevant_chunks(
    documents: List[Tuple[str, str]],
    size_threshold_words: int = SIZE_THRESHOLD_WORDS,
    top_k_per_query: int = TOP_K_PER_QUERY,
) -> Tuple[str, dict]:
    """
    documents: list of (label, text) pairs, one per input document.
    Returns (selected_text, stats) where selected_text is ready to feed
    directly into llm_engine.ask_chatgpt, and stats records whether
    retrieval activated and how much of the corpus was kept (for
    transparency and for the eval harness).
    """
    total_words = sum(_word_count(t) for _, t in documents)

    # Below threshold: concatenate everything, no retrieval/embedding calls.
    if total_words <= size_threshold_words:
        selected_text = "\n\n".join(f"[Document: {label}]\n{text}" for label, text in documents)
        return selected_text, {
            "retrieval_activated": False,
            "total_words": total_words,
            "total_chunks": None,
            "selected_chunks": None,
        }

    # Build the full chunk list, tagged with source document label.
    tagged_chunks: List[Tuple[str, str]] = []  # (label, chunk_text)
    for label, text in documents:
        for chunk in chunk_document(text):
            tagged_chunks.append((label, chunk))

    if not tagged_chunks:
        return "", {"retrieval_activated": True, "total_words": total_words, "total_chunks": 0, "selected_chunks": 0}

    chunk_texts = [c for _, c in tagged_chunks]
    all_embeddings = embed_texts(RETRIEVAL_QUERIES + chunk_texts)
    query_embeddings = all_embeddings[: len(RETRIEVAL_QUERIES)]
    chunk_embeddings = all_embeddings[len(RETRIEVAL_QUERIES):]

    selected_indices = set()
    for q_emb in query_embeddings:
        scored = sorted(
            range(len(chunk_embeddings)),
            key=lambda i: cosine_similarity(q_emb, chunk_embeddings[i]),
            reverse=True,
        )
        selected_indices.update(scored[:top_k_per_query])

    # Preserve original document/chunk order for readability, not score order.
    ordered_selected = sorted(selected_indices)
    selected_text = "\n\n".join(
        f"[Document: {tagged_chunks[i][0]}]\n{tagged_chunks[i][1]}" for i in ordered_selected
    )

    return selected_text, {
        "retrieval_activated": True,
        "total_words": total_words,
        "total_chunks": len(tagged_chunks),
        "selected_chunks": len(ordered_selected),
    }
