"""API-facing request/response schemas (distinct from internal models so the
public contract can evolve independently of internal representations)."""
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    qdrant_available: bool
    redis_available: bool


class CacheClearResponse(BaseModel):
    entries_cleared: int


class CacheThresholdUpdateRequest(BaseModel):
    threshold: float


class DocumentDeleteResponse(BaseModel):
    document_name: str
    deleted: bool
