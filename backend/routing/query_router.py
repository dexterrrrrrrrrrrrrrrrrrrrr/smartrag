"""
Query complexity router.

Deliberately a heuristic, explainable scorer rather than an ML classifier —
the router's job here is to be interview-explainable: every signal that
goes into the decision is visible in `ComplexitySignals`, and the final
label is a simple threshold on their weighted sum. Swapping this for a
learned classifier later only requires implementing the same
`QueryRouter.route()` interface (see README "Future improvements").
"""
from backend.models.llm import ModelRole
from backend.models.routing import ComplexitySignals, RouteDecision

_COMPARISON_KEYWORDS = (
    "compare", "comparison", "versus", " vs ", "difference between",
    "advantages and disadvantages", "pros and cons", "better than",
)
_MULTI_STEP_KEYWORDS = (
    "step by step", "first,", "then,", "explain how", "walk me through",
    "and also", "as well as", "additionally", "furthermore",
)

# Weights are intentionally simple and documented so the scoring is
# auditable — not a black box.
_WEIGHTS = {
    "length": 0.25,       # long queries tend to need more reasoning
    "questions": 0.15,     # multiple questions in one query = more work
    "comparison": 0.30,    # comparative reasoning is inherently multi-part
    "multi_step": 0.20,    # explicit multi-step / sequential asks
    "chunks": 0.10,        # more retrieved context to synthesize = harder
}

# A query scoring at or above this is routed to the LARGE model.
COMPLEXITY_THRESHOLD = 0.45


class QueryRouter:
    def __init__(self, threshold: float = COMPLEXITY_THRESHOLD):
        self.threshold = threshold

    def _score_length(self, word_count: int) -> float:
        # Normalize: 0 words -> 0.0, 40+ words -> 1.0
        return min(word_count / 40.0, 1.0)

    def _score_questions(self, question_marks: int) -> float:
        # 1 question -> 0.0 contribution, 2+ -> escalating
        return min(max(question_marks - 1, 0) / 2.0, 1.0)

    def route(self, query: str, num_retrieved_chunks: int = 0) -> RouteDecision:
        lowered = query.lower()
        word_count = len(query.split())
        question_marks = query.count("?")
        has_comparison = any(kw in lowered for kw in _COMPARISON_KEYWORDS)
        has_multi_step = any(kw in lowered for kw in _MULTI_STEP_KEYWORDS)

        length_score = self._score_length(word_count)
        question_score = self._score_questions(question_marks)
        comparison_score = 1.0 if has_comparison else 0.0
        multi_step_score = 1.0 if has_multi_step else 0.0
        chunks_score = min(num_retrieved_chunks / 8.0, 1.0)

        score = (
            _WEIGHTS["length"] * length_score
            + _WEIGHTS["questions"] * question_score
            + _WEIGHTS["comparison"] * comparison_score
            + _WEIGHTS["multi_step"] * multi_step_score
            + _WEIGHTS["chunks"] * chunks_score
        )

        signals = ComplexitySignals(
            char_length=len(query),
            word_count=word_count,
            question_marks=question_marks,
            has_comparison_keywords=has_comparison,
            has_multi_step_keywords=has_multi_step,
            num_retrieved_chunks=num_retrieved_chunks,
            score=round(score, 4),
        )

        if score >= self.threshold:
            reasons = []
            if has_comparison:
                reasons.append("comparative reasoning requested")
            if has_multi_step:
                reasons.append("multi-step / sequential reasoning requested")
            if question_score > 0:
                reasons.append("multiple questions in one query")
            if length_score > 0.5:
                reasons.append("long, detailed query")
            if chunks_score > 0.5:
                reasons.append("large amount of retrieved context to synthesize")
            reason = "; ".join(reasons) or f"complexity score {score:.2f} >= threshold {self.threshold}"
            return RouteDecision(
                role=ModelRole.LARGE, complexity_label="HIGH", reason=reason, signals=signals
            )

        return RouteDecision(
            role=ModelRole.SMALL,
            complexity_label="LOW",
            reason="Simple, short, single-part factual question",
            signals=signals,
        )
