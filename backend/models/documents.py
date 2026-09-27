"""Schemas for document ingestion and chunking."""
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_name: str
    page_number: int | None = None
    chunk_id: str
    source: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentChunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class IngestionResult(BaseModel):
    document_name: str
    num_chunks: int
    total_characters: int
    latency_ms: float
