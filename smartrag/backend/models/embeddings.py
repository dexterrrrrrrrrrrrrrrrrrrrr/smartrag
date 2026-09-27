"""Shared schemas for embedding operations."""
from pydantic import BaseModel


class EmbeddingResult(BaseModel):
    vector: list[float]
    dimension: int
    model_name: str
    latency_ms: float
