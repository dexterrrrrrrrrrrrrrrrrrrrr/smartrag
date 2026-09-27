"""
Dependency providers for FastAPI routes.

Services that need a persistent connection (Qdrant, Redis) are constructed
once and reused (module-level singletons) rather than reconnecting per
request. Construction failures are deferred to first use inside each
service (see QdrantStore/SemanticCache), so the API can still start up even
if Qdrant or Redis happen to be down at boot — individual requests will
then surface a clear 503 instead of the whole app failing to launch.
"""
from functools import lru_cache

from backend.analytics.db import get_session
from backend.analytics.repository import AnalyticsRepository
from backend.cache.semantic_cache import SemanticCache
from backend.core.config import Settings, get_settings
from backend.embeddings.embedder import Embedder
from backend.llm.ollama_client import OllamaClient
from backend.rag.pipeline import RAGPipeline
from backend.retrieval.ingestion_service import IngestionService
from backend.retrieval.qdrant_store import QdrantStore
from backend.routing.query_router import QueryRouter


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_settings())


@lru_cache
def get_llm_client() -> OllamaClient:
    return OllamaClient(get_settings())


@lru_cache
def get_semantic_cache() -> SemanticCache:
    return SemanticCache.from_settings(get_settings())


@lru_cache
def get_qdrant_store() -> QdrantStore:
    return QdrantStore.from_settings(get_settings())


@lru_cache
def get_query_router() -> QueryRouter:
    return QueryRouter()


def get_analytics_repository() -> AnalyticsRepository:
    """Not cached — a fresh SQLModel session per request, closed by the
    route handler via a context manager."""
    return AnalyticsRepository(get_session(get_settings()))


def get_ingestion_service() -> IngestionService:
    return IngestionService(get_embedder(), get_qdrant_store())


def get_rag_pipeline(analytics: AnalyticsRepository) -> RAGPipeline:
    return RAGPipeline(
        embedder=get_embedder(),
        cache=get_semantic_cache(),
        router=get_query_router(),
        store=get_qdrant_store(),
        llm=get_llm_client(),
        analytics=analytics,
        settings=get_settings(),
    )
