"""Top-level response schema returned by the RAG pipeline / API."""
from pydantic import BaseModel

from backend.models.cache import SourceCitation
from backend.models.routing import RouteDecision


class LatencyBreakdown(BaseModel):
    embedding_ms: float
    cache_lookup_ms: float
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float


class RAGAnswer(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]

    cache_hit: bool
    similarity_score: float | None = None

    route: RouteDecision | None = None  # None when served from cache
    num_chunks_retrieved: int = 0

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    latency: LatencyBreakdown
    estimated_cost_usd: float
    estimated_cost_if_no_cache_usd: float
