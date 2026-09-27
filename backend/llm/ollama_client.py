"""
Thin wrapper around the Ollama HTTP API.

Design goals:
- Callers ask for a `ModelRole` (SMALL / LARGE), never a literal model name.
  The actual model name always comes from Settings (`.env`), so swapping
  models never requires touching this file or any caller.
- Every failure mode Ollama can produce (server down, model missing, timeout)
  is translated into a specific, actionable exception rather than a raw
  httpx error, so the API layer can return clean error messages.
- No hidden global state: pass a Settings instance in, get a client out.
"""
import time

import httpx

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.llm.exceptions import (
    OllamaGenerationError,
    OllamaModelNotFoundError,
    OllamaUnavailableError,
)
from backend.models.llm import GenerationResult, ModelRole

logger = get_logger(__name__)


class OllamaClient:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._role_to_model = {
            ModelRole.SMALL: self.settings.small_model,
            ModelRole.LARGE: self.settings.large_model,
        }

    def resolve_model_name(self, role: ModelRole) -> str:
        """Map a logical role to the concrete model name from config."""
        return self._role_to_model[role]

    def is_available(self) -> bool:
        """Quick health check — used at startup and by API health endpoints."""
        try:
            resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=3.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def list_installed_models(self) -> list[str]:
        try:
            resp = httpx.get(f"{self.settings.ollama_base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except httpx.HTTPError as exc:
            raise OllamaUnavailableError(self.settings.ollama_base_url) from exc

    def ensure_model_available(self, role: ModelRole) -> None:
        """Raise a clear, actionable error if the role's model isn't pulled yet."""
        model_name = self.resolve_model_name(role)
        installed = self.list_installed_models()
        # Ollama tags include version suffixes (e.g. "qwen2.5:3b"); match loosely
        # against the configured name so "qwen2.5:3b" matches "qwen2.5:3b" exactly
        # and also tolerates a trailing ":latest" style mismatch.
        if model_name not in installed and f"{model_name}:latest" not in installed:
            raise OllamaModelNotFoundError(model_name)

    def generate(
        self,
        prompt: str,
        role: ModelRole,
        system: str | None = None,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Call Ollama's /api/generate for the given role and return a
        structured result, including latency for observability."""
        model_name = self.resolve_model_name(role)
        payload = {
            "model": model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"temperature": temperature},
        }

        start = time.perf_counter()
        try:
            resp = httpx.post(
                f"{self.settings.ollama_base_url}/api/generate",
                json=payload,
                timeout=self.settings.llm_request_timeout_seconds,
            )
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(self.settings.ollama_base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaGenerationError(
                f"Ollama request timed out after {self.settings.llm_request_timeout_seconds}s "
                f"for model '{model_name}'."
            ) from exc

        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 404:
            raise OllamaModelNotFoundError(model_name)
        if resp.status_code != 200:
            raise OllamaGenerationError(
                f"Ollama returned HTTP {resp.status_code} for model '{model_name}': {resp.text[:300]}"
            )

        data = resp.json()
        logger.debug(f"Ollama generation via {model_name} took {latency_ms:.1f}ms")

        return GenerationResult(
            text=data.get("response", ""),
            model_name=model_name,
            model_role=role,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            total_tokens=(data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0)
            if data.get("eval_count") is not None
            else None,
            latency_ms=latency_ms,
        )
