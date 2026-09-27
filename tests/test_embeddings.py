"""
Tests for the local embedding wrapper. These run fully offline once the
sentence-transformers model has been downloaded once (cached by HuggingFace).
"""
import pytest

from backend.embeddings.embedder import Embedder


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def test_embed_returns_correct_dimension(embedder: Embedder):
    result = embedder.embed("What is RAG?")
    assert result.dimension == embedder.settings.embedding_dimension
    assert len(result.vector) == result.dimension


def test_embed_batch_matches_single(embedder: Embedder):
    texts = ["What is RAG?", "How do I implement authentication in FastAPI?"]
    batch_results = embedder.embed_batch(texts)
    assert len(batch_results) == 2
    for r in batch_results:
        assert r.dimension == embedder.settings.embedding_dimension


def test_similar_queries_have_high_similarity(embedder: Embedder):
    a = embedder.embed("What is RAG?")
    b = embedder.embed("Can you explain what RAG means?")
    sim = Embedder.cosine_similarity(a.vector, b.vector)
    assert sim > 0.5  # loosely similar in meaning


def test_dissimilar_queries_have_lower_similarity(embedder: Embedder):
    a = embedder.embed("What is RAG?")
    b = embedder.embed("How do I implement authentication in FastAPI?")
    sim = Embedder.cosine_similarity(a.vector, b.vector)

    c = embedder.embed("Can you explain what RAG means?")
    sim_related = Embedder.cosine_similarity(a.vector, c.vector)

    assert sim < sim_related


def test_cosine_similarity_identical_vectors_is_one():
    v = [1.0, 0.0, 0.0]
    assert Embedder.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert Embedder.cosine_similarity(a, b) == pytest.approx(0.0)
