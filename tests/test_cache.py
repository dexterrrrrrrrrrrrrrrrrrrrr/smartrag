"""Tests for the semantic cache: hits, misses, threshold behavior, and the
critical safety property that semantically-related-but-different-intent
queries must NOT share a cached answer."""
from backend.cache.semantic_cache import SemanticCache
from backend.models.cache import CacheEntry
from backend.models.llm import ModelRole


def _make_entry(query: str, embedding: list[float], answer: str) -> CacheEntry:
    return CacheEntry(
        entry_id=SemanticCache.new_entry_id(),
        original_query=query,
        query_embedding=embedding,
        answer=answer,
        sources=[],
        model_used=ModelRole.SMALL,
        model_name="test-small:1b",
        total_tokens=100,
    )


def test_exact_cache_hit(semantic_cache):
    emb = [1.0, 0.0, 0.0]
    entry = _make_entry("What is RAG?", emb, "RAG combines retrieval and generation.")
    semantic_cache.store(entry)

    result = semantic_cache.lookup(emb)  # identical embedding = exact match

    assert result.hit is True
    assert result.similarity_score == 1.0
    assert result.entry.answer == "RAG combines retrieval and generation."


def test_semantic_cache_hit_on_paraphrase(semantic_cache):
    emb_original = [1.0, 0.0, 0.0]
    semantic_cache.store(_make_entry("What is Retrieval Augmented Generation?", emb_original, "Answer A"))

    # A paraphrase's embedding won't be byte-identical but should be highly similar
    emb_paraphrase = [0.98, 0.199, 0.0]
    result = semantic_cache.lookup(emb_paraphrase)

    assert result.hit is True
    assert result.similarity_score >= semantic_cache.similarity_threshold


def test_cache_miss_on_unrelated_query(semantic_cache):
    semantic_cache.store(_make_entry("What is RAG?", [1.0, 0.0, 0.0], "Answer A"))

    unrelated_emb = [0.0, 1.0, 0.0]  # orthogonal -> similarity 0
    result = semantic_cache.lookup(unrelated_emb)

    assert result.hit is False
    assert result.entry is None


def test_cache_miss_when_cache_is_empty(semantic_cache):
    result = semantic_cache.lookup([1.0, 0.0, 0.0])
    assert result.hit is False
    assert result.candidates_checked == 0


def test_cache_threshold_boundary(fake_redis_client):
    """A similarity just below threshold must miss; just above must hit."""
    cache = SemanticCache(fake_redis_client, similarity_threshold=0.90)
    cache.store(_make_entry("query", [1.0, 0.0], "answer"))

    import math

    # cos(theta) = 0.89 -> construct a vector at that similarity
    below = [0.89, math.sqrt(1 - 0.89**2)]
    above = [0.95, math.sqrt(1 - 0.95**2)]

    result_below = cache.lookup(below)
    result_above = cache.lookup(above)

    assert result_below.hit is False
    assert result_above.hit is True


def test_cache_threshold_is_configurable_at_runtime(fake_redis_client):
    cache = SemanticCache(fake_redis_client, similarity_threshold=0.99)
    cache.store(_make_entry("query", [1.0, 0.0], "answer"))

    similar_but_not_identical = [0.95, 0.312]  # ~0.95 similarity
    assert cache.lookup(similar_but_not_identical).hit is False

    cache.similarity_threshold = 0.90  # dashboard-style runtime update
    assert cache.lookup(similar_but_not_identical).hit is True


def test_different_intent_not_incorrectly_cached(semantic_cache):
    """'Capital of France' vs 'population of France' are topically related
    (share the word 'France') but ask fundamentally different questions.
    Their embeddings should NOT be similar enough to trigger a false hit."""
    semantic_cache.store(_make_entry("What is the capital of France?", [1.0, 0.0, 0.0], "Paris."))

    # Related topic, different intent -> moderate similarity, must stay below threshold
    population_query_emb = [0.75, 0.6614, 0.0]
    result = semantic_cache.lookup(population_query_emb)

    assert result.hit is False
    assert result.entry is None


def test_cache_invalidate_removes_entry(semantic_cache):
    entry = _make_entry("What is RAG?", [1.0, 0.0, 0.0], "Answer A")
    semantic_cache.store(entry)
    assert semantic_cache.stats()["num_entries"] == 1

    removed = semantic_cache.invalidate(entry.entry_id)
    assert removed is True
    assert semantic_cache.stats()["num_entries"] == 0
    assert semantic_cache.lookup([1.0, 0.0, 0.0]).hit is False


def test_cache_clear_removes_all_entries(semantic_cache):
    semantic_cache.store(_make_entry("q1", [1.0, 0.0, 0.0], "a1"))
    semantic_cache.store(_make_entry("q2", [0.0, 1.0, 0.0], "a2"))
    assert semantic_cache.stats()["num_entries"] == 2

    cleared = semantic_cache.clear()
    assert cleared == 2
    assert semantic_cache.stats()["num_entries"] == 0
