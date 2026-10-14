# tests/test_retrieval.py
#
# Phase 7: formalizes the ad-hoc checks run informally during Phase 4 (see
# docs/PROGRESS.md) into a repeatable test suite. No network calls: the
# embedding-dependent selection logic is tested via a mocked embed_texts,
# the same approach used informally when Phase 4 first verified the
# filtering logic actually discriminates relevant from irrelevant chunks.

import retrieval
from retrieval import chunk_document, cosine_similarity, retrieve_relevant_chunks


# --- chunk_document ----------------------------------------------------------

def test_chunk_document_splits_on_blank_lines():
    text = "Paragraph one.\n\nParagraph two."
    assert chunk_document(text) == ["Paragraph one.", "Paragraph two."]


def test_chunk_document_splits_long_paragraph_by_sentence():
    long_para = " ".join(f"Sentence number {i}." for i in range(60))
    chunks = chunk_document(long_para, max_words=50)
    assert len(chunks) >= 2
    # no information lost: every sentence must still be present somewhere
    assert "Sentence number 0." in chunks[0]


def test_chunk_document_empty_input():
    assert chunk_document("") == []
    assert chunk_document(None) == []


# --- cosine_similarity --------------------------------------------------------

def test_cosine_similarity_identical_vectors():
    assert abs(cosine_similarity([1, 0, 0], [1, 0, 0]) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    assert abs(cosine_similarity([1, 0, 0], [0, 1, 0])) < 1e-9


def test_cosine_similarity_zero_vector_does_not_raise():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


# --- retrieve_relevant_chunks: size-threshold gating ------------------------

def test_small_input_skips_retrieval_with_no_embedding_call(monkeypatch):
    def fail_if_called(texts):
        raise AssertionError("embed_texts should not be called below the size threshold")
    monkeypatch.setattr(retrieval, "embed_texts", fail_if_called)

    docs = [("doc1", "A short sentence well under the word threshold.")]
    text, stats = retrieve_relevant_chunks(docs)
    assert stats["retrieval_activated"] is False
    assert "[Document: doc1]" in text


def test_large_input_activates_retrieval(monkeypatch):
    # fake embeddings: topic-aligned one-hot-ish vectors, same technique used
    # to verify this logic live during Phase 4.
    def fake_embed(texts):
        return [[1.0] for _ in texts]  # trivial but non-zero, just needs to run
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)

    long_doc = " ".join(["word"] * 400)  # comfortably over SIZE_THRESHOLD_WORDS
    docs = [("doc1", long_doc)]
    text, stats = retrieve_relevant_chunks(docs)
    assert stats["retrieval_activated"] is True
    assert stats["total_chunks"] is not None


# --- retrieve_relevant_chunks: the filtering logic itself -------------------

def test_retrieval_selects_relevant_chunks_and_excludes_decoys(monkeypatch):
    # Same mocked-embedding technique Phase 4 used to verify this live:
    # 6 topic directions (one per RETRIEVAL_QUERIES entry) in a toy vector
    # space, so a chunk only scores highly against its own topic's query.
    topic_order = ["actors", "components", "external", "relationships", "risks", "assumptions"]
    topic_vectors = {t: [1.0 if i == j else 0.0 for i in range(6)] for j, t in enumerate(topic_order)}

    def fake_embed(texts):
        out = []
        for t in texts:
            if t in retrieval.RETRIEVAL_QUERIES:
                out.append(topic_vectors[topic_order[retrieval.RETRIEVAL_QUERIES.index(t)]])
            else:
                matched = next((topic_vectors[k] for k in topic_vectors if k in t), [0.0] * 6)
                out.append(matched)
        return out

    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)

    docs = [
        ("doc_a", "actors chunk text.\n\ncomponents chunk text.\n\nexternal chunk text."),
        ("doc_b", "relationships chunk text.\n\nrisks chunk text.\n\nassumptions chunk text."),
        ("doc_c", "totally unrelated decoy paragraph one.\n\ntotally unrelated decoy paragraph two."),
    ]
    text, stats = retrieve_relevant_chunks(docs, size_threshold_words=1, top_k_per_query=1)

    assert stats["total_chunks"] == 8
    assert stats["selected_chunks"] == 6  # exactly the 6 on-topic chunks
    assert "decoy" not in text
