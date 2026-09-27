"""Tests for cost estimation — must be computed from real token counts times
the configured reference price, never a hardcoded constant."""
import pytest

from backend.analytics.cost import estimate_avoided_cost_usd, estimate_cost_usd
from backend.core.config import Settings
from backend.models.llm import ModelRole


def test_cost_scales_linearly_with_tokens():
    settings = Settings(small_model_reference_cost_per_1k_tokens=0.0002)
    cost_100 = estimate_cost_usd(100, ModelRole.SMALL, settings)
    cost_200 = estimate_cost_usd(200, ModelRole.SMALL, settings)
    assert cost_200 == pytest.approx(cost_100 * 2)


def test_cost_uses_correct_price_per_role():
    settings = Settings(
        small_model_reference_cost_per_1k_tokens=0.0002,
        large_model_reference_cost_per_1k_tokens=0.0020,
    )
    small_cost = estimate_cost_usd(1000, ModelRole.SMALL, settings)
    large_cost = estimate_cost_usd(1000, ModelRole.LARGE, settings)

    assert small_cost == 0.0002
    assert large_cost == 0.0020
    assert large_cost > small_cost  # large model must be priced higher in the reference table


def test_zero_tokens_costs_zero():
    settings = Settings()
    assert estimate_cost_usd(0, ModelRole.SMALL, settings) == 0.0


def test_avoided_cost_matches_would_be_generation_cost():
    settings = Settings(large_model_reference_cost_per_1k_tokens=0.0020)
    avoided = estimate_avoided_cost_usd(1000, ModelRole.LARGE, settings)
    direct = estimate_cost_usd(1000, ModelRole.LARGE, settings)
    assert avoided == direct


def test_cost_pricing_is_configurable_not_hardcoded():
    """Changing the .env-equivalent Settings value must change the computed
    cost — proves the price isn't hardcoded in the calculation function."""
    cheap_settings = Settings(small_model_reference_cost_per_1k_tokens=0.0001)
    expensive_settings = Settings(small_model_reference_cost_per_1k_tokens=0.01)

    cheap_cost = estimate_cost_usd(1000, ModelRole.SMALL, cheap_settings)
    expensive_cost = estimate_cost_usd(1000, ModelRole.SMALL, expensive_settings)

    assert expensive_cost > cheap_cost
