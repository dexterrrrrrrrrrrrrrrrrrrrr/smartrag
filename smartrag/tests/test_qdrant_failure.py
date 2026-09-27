"""Tests that Qdrant connection failures are translated into a clear,
typed error rather than a raw exception leaking to the API layer."""
import pytest

from backend.core.config import Settings
from backend.retrieval.qdrant_store import QdrantStore, QdrantUnavailableError


def test_qdrant_store_from_settings_raises_typed_error_when_unreachable():
    # Nothing is listening on this port, so connection must fail fast.
    settings = Settings(qdrant_url="http://localhost:1")
    with pytest.raises(QdrantUnavailableError):
        QdrantStore.from_settings(settings)


def test_qdrant_unavailable_error_message_is_actionable():
    settings = Settings(qdrant_url="http://localhost:1")
    try:
        QdrantStore.from_settings(settings)
        pytest.fail("Expected QdrantUnavailableError")
    except QdrantUnavailableError as exc:
        assert "docker compose up" in str(exc)
