"""
Estimated API-equivalent cost calculation.

IMPORTANT: This application runs entirely on local, free infrastructure.
No real money is ever spent. The numbers produced here answer the
hypothetical question "what would this have cost against a metered cloud
API at the reference prices in .env?" — always computed from real token
counts returned by Ollama, never fabricated or hardcoded per-call.
"""
from backend.core.config import Settings, get_settings
from backend.models.llm import ModelRole


def estimate_cost_usd(
    total_tokens: int,
    role: ModelRole,
    settings: Settings | None = None,
) -> float:
    """Estimated / hypothetical API-equivalent cost for a real generation.
    Returns 0.0 for cache hits (no tokens were generated) — callers should
    not call this at all for cache hits; use `estimate_cost_if_generated`
    instead to compute the "what it would have cost without caching" figure.
    """
    settings = settings or get_settings()
    price_per_1k = (
        settings.small_model_reference_cost_per_1k_tokens
        if role == ModelRole.SMALL
        else settings.large_model_reference_cost_per_1k_tokens
    )
    return round((total_tokens / 1000.0) * price_per_1k, 8)


def estimate_avoided_cost_usd(
    would_be_total_tokens: int,
    would_be_role: ModelRole,
    settings: Settings | None = None,
) -> float:
    """For a cache HIT: estimate what this query *would have* cost if it had
    required a fresh LLM call. `would_be_total_tokens` should be a
    reasonable estimate — in this app we use the token count actually
    recorded on the cached entry's original (cache-populating) generation,
    since that's a real, previously-measured number rather than a guess."""
    return estimate_cost_usd(would_be_total_tokens, would_be_role, settings)
