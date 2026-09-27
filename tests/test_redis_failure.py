"""Tests that the semantic cache degrades gracefully (always-miss, never
crashes) when Redis is unreachable, per the requirement that the app keeps
functioning as much as possible if Redis is down."""
import redis

from backend.cache.semantic_cache import SemanticCache
from backend.models.cache import CacheEntry
from backend.models.llm import ModelRole


def _unreachable_redis_client() -> redis.Redis:
    # Nothing listens on port 1 — guaranteed connection refusal, fast timeout.
    return redis.Redis(host="localhost", port=1, socket_connect_timeout=0.5, socket_timeout=0.5)


def test_is_available_false_when_redis_down():
    cache = SemanticCache(_unreachable_redis_client(), similarity_threshold=0.9)
    assert cache.is_available() is False


def test_lookup_returns_graceful_miss_when_redis_down():
    cache = SemanticCache(_unreachable_redis_client(), similarity_threshold=0.9)
    result = cache.lookup([1.0, 0.0, 0.0])
    assert result.hit is False  # must not raise


def test_store_does_not_raise_when_redis_down():
    cache = SemanticCache(_unreachable_redis_client(), similarity_threshold=0.9)
    entry = CacheEntry(
        entry_id="x",
        original_query="q",
        query_embedding=[1.0, 0.0, 0.0],
        answer="a",
        sources=[],
        model_used=ModelRole.SMALL,
        model_name="test-small:1b",
    )
    cache.store(entry)  # should log a warning internally, not raise


def test_stats_reports_unavailable_when_redis_down():
    cache = SemanticCache(_unreachable_redis_client(), similarity_threshold=0.9)
    stats = cache.stats()
    assert stats["available"] is False
