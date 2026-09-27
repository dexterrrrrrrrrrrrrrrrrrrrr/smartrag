"""
Shared pytest fixtures.

`fake_embedder` provides a deterministic, network-free stand-in for the real
sentence-transformers `Embedder` so most tests can run in CI/offline
environments without downloading model weights. Tests that specifically
need to validate the *real* embedding model (see test_embeddings.py) use
the real `Embedder` directly and are skipped automatically if it can't be
loaded (e.g. no internet on first run).
"""
import hashlib

import fakeredis
import numpy as np
import pytest
from qdrant_client import QdrantClient

from backend.cache.semantic_cache import SemanticCache
from backend.core.config import Settings
from backend.models.embeddings import EmbeddingResult
from backend.retrieval.qdrant_store import QdrantStore
from backend.routing.query_router import QueryRouter


class FakeEmbedder:
    """Deterministic bag-of-words style embedding: same text always maps to
    the same vector, semantically related word overlap increases cosine
    similarity, and it needs no network access or model download."""

    DIMENSION = 32

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings(embedding_dimension=self.DIMENSION)
        self.model_name = "fake-deterministic-embedder"

    def embed(self, text: str) -> EmbeddingResult:
        vec = np.zeros(self.DIMENSION)
        for word in text.lower().split():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vec[h % self.DIMENSION] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return EmbeddingResult(
            vector=vec.tolist(), dimension=self.DIMENSION, model_name=self.model_name, latency_ms=0.1
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        return [self.embed(t) for t in texts]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        arr_a, arr_b = np.array(a), np.array(b)
        na, nb = np.linalg.norm(arr_a), np.linalg.norm(arr_b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(arr_a, arr_b) / (na * nb))


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def in_memory_qdrant_store() -> QdrantStore:
    client = QdrantClient(location=":memory:")
    return QdrantStore(client, collection_name="test_collection", embedding_dimension=FakeEmbedder.DIMENSION)


@pytest.fixture
def fake_redis_client():
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def semantic_cache(fake_redis_client) -> SemanticCache:
    return SemanticCache(fake_redis_client, similarity_threshold=0.90, ttl_seconds=0)


@pytest.fixture
def query_router() -> QueryRouter:
    return QueryRouter()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        small_model="test-small:1b",
        large_model="test-large:7b",
        embedding_dimension=FakeEmbedder.DIMENSION,
        cache_similarity_threshold=0.90,
    )
