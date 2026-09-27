"""Tests that Ollama failures (server down, model missing, timeout) are
translated into specific, actionable exceptions."""
import pytest

from backend.core.config import Settings
from backend.llm.exceptions import OllamaModelNotFoundError, OllamaUnavailableError
from backend.llm.ollama_client import OllamaClient
from backend.models.llm import ModelRole


@pytest.fixture
def unreachable_client() -> OllamaClient:
    return OllamaClient(Settings(ollama_base_url="http://localhost:1"))


def test_is_available_false_when_ollama_down(unreachable_client):
    assert unreachable_client.is_available() is False


def test_list_models_raises_when_ollama_down(unreachable_client):
    with pytest.raises(OllamaUnavailableError):
        unreachable_client.list_installed_models()


def test_generate_raises_unavailable_when_ollama_down(unreachable_client):
    with pytest.raises(OllamaUnavailableError):
        unreachable_client.generate("hello", role=ModelRole.SMALL)


def test_ensure_model_available_raises_when_model_missing(monkeypatch):
    client = OllamaClient(Settings(small_model="does-not-exist:1b"))
    monkeypatch.setattr(client, "list_installed_models", lambda: ["some-other-model:7b"])
    with pytest.raises(OllamaModelNotFoundError):
        client.ensure_model_available(ModelRole.SMALL)


def test_ensure_model_available_passes_when_model_present(monkeypatch):
    client = OllamaClient(Settings(small_model="qwen2.5:3b"))
    monkeypatch.setattr(client, "list_installed_models", lambda: ["qwen2.5:3b", "qwen2.5:7b"])
    client.ensure_model_available(ModelRole.SMALL)  # should not raise


def test_error_messages_are_actionable():
    settings = Settings(ollama_base_url="http://localhost:1")
    client = OllamaClient(settings)
    try:
        client.list_installed_models()
        pytest.fail("expected OllamaUnavailableError")
    except OllamaUnavailableError as exc:
        assert "ollama serve" in str(exc)

    model_err = OllamaModelNotFoundError("qwen2.5:3b")
    assert "ollama pull qwen2.5:3b" in str(model_err)
