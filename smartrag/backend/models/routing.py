"""Schemas for the query complexity router."""
from pydantic import BaseModel

from backend.models.llm import ModelRole


class ComplexitySignals(BaseModel):
    """Explainable, individually-inspectable signals that feed the routing
    decision. Kept as a distinct object (rather than folded straight into a
    score) so the API/dashboard can show *why* a query was routed a
    particular way."""

    char_length: int
    word_count: int
    question_marks: int
    has_comparison_keywords: bool
    has_multi_step_keywords: bool
    num_retrieved_chunks: int
    score: float


class RouteDecision(BaseModel):
    role: ModelRole
    complexity_label: str  # "LOW" | "HIGH"
    reason: str
    signals: ComplexitySignals
