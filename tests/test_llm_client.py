"""
Tests for the Ollama client wrapper.

`test_resolve_model_name_*` and the unavailable-server test run fully
offline. `test_live_generation_*` are marked and skipped automatically if
Ollama isn't running, so `pytest` still passes in a fresh checkout before
the user has started Ollama.
"""
import httpx
import pytest

from backend.core.config import Settings
from backend.llm.exceptions import OllamaUnavailableError
from backend.llm.ollama_client import OllamaClient
from backend.models.llm import ModelRole


@pytest.fixture
def settings() -> Settings:
    return Settings(small_model="test-small:1b", large_model="test-large:7b")


def test_resolve_model_name_small(settings: Settings):
    client = OllamaClient(settings)
    assert client.resolve_model_name(ModelRole.SMALL) == "test-small:1b"


def test_resolve_model_name_large(settings: Settings):
    client = OllamaClient(settings)
    assert client.resolve_model_name(ModelRole.LARGE) == "test-large:7b"


def test_is_available_false_when_server_unreachable():
    settings = Settings(ollama_base_url="http://localhost:1")  # nothing listens here
    client = OllamaClient(settings)
    assert client.is_available() is False


def test_list_installed_models_raises_when_unreachable():
    settings = Settings(ollama_base_url="http://localhost:1")
    client = OllamaClient(settings)
    with pytest.raises(OllamaUnavailableError):
        client.list_installed_models()


def _ollama_running(base_url: str = "http://localhost:11434") -> bool:
    try:
        return httpx.get(f"{base_url}/api/tags", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not _ollama_running(), reason="Ollama is not running locally")
def test_live_ollama_health_check():
    client = OllamaClient()
    assert client.is_available() is True


@pytest.mark.skipif(not _ollama_running(), reason="Ollama is not running locally")
def test_live_generation_with_small_model():
    client = OllamaClient()
    installed = client.list_installed_models()
    small = client.settings.small_model
    if small not in installed and f"{small}:latest" not in installed:
        pytest.skip(f"{small} not pulled locally yet")
    result = client.generate("Say 'hello' and nothing else.", role=ModelRole.SMALL)
    assert result.text
    assert result.latency_ms > 0
