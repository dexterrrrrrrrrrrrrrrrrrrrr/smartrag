"""Schemas for the semantic cache layer."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from backend.models.llm import ModelRole


class SourceCitation(BaseModel):
    document_name: str
    page_number: int | None = None


class CacheEntry(BaseModel):
    """What gets stored in Redis for every answered (non-cached) query."""

    entry_id: str
    original_query: str
    query_embedding: list[float]
    answer: str
    sources: list[SourceCitation]
    model_used: ModelRole
    model_name: str
    # Real token counts from the original generation that populated this
    # cache entry — used to compute "cost avoided" for future hits against
    # actual historical usage, not a guess.
    total_tokens: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CacheLookupResult(BaseModel):
    hit: bool
    entry: CacheEntry | None = None
    similarity_score: float | None = None
    lookup_latency_ms: float
    candidates_checked: int = 0
