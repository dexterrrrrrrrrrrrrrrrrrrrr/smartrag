"""Shared schemas for LLM interactions."""
from enum import Enum

from pydantic import BaseModel


class ModelRole(str, Enum):
    """Logical model roles. Never reference a literal model name outside
    of Settings — always go through a role and let config resolve it."""

    SMALL = "small"
    LARGE = "large"


class GenerationResult(BaseModel):
    text: str
    model_name: str
    model_role: ModelRole
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float
