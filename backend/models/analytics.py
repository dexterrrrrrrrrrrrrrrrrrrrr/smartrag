"""SQLModel table + schemas for per-request analytics logging."""
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class RequestLog(SQLModel, table=True):
    """One row per query handled by the RAG pipeline. This is the single
    source of truth the dashboard, benchmark script, and cost calculations
    read from — nothing is computed from fabricated numbers."""

    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    query: str

    # Cache outcome
    cache_hit: bool
    similarity_score: float | None = None

    # Routing outcome (null when served entirely from cache, since no
    # generation happened)
    model_role: str | None = None  # "small" | "large"
    model_name: str | None = None
    complexity_label: str | None = None
    complexity_score: float | None = None

    # Retrieval
    num_chunks_retrieved: int = 0

    # Tokens (only known when an LLM call actually happened)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # Latency breakdown (ms) — every field here is a real measurement
    embedding_latency_ms: float = 0.0
    cache_lookup_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Cost — hypothetical/reference only, computed from real token counts *
    # the configured reference price. Zero real money is ever involved.
    estimated_cost_usd: float = 0.0
    # What this query *would have* cost if caching were disabled (i.e. if it
    # had needed a fresh LLM call at reference pricing) — lets the dashboard
    # compute "cost avoided" for cache hits.
    estimated_cost_if_no_cache_usd: float = 0.0

    error: str | None = None
