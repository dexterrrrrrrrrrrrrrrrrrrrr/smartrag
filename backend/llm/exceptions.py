"""Exceptions for the LLM layer, so callers (API routes, RAG pipeline) can
catch specific, actionable failure modes instead of a bare Exception."""


class OllamaError(Exception):
    """Base class for all Ollama-related failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama server cannot be reached at all."""

    def __init__(self, base_url: str):
        super().__init__(
            f"Could not reach Ollama at {base_url}. "
            "Is it running? Try: `ollama serve` (or open the Ollama app), "
            "then retry."
        )
        self.base_url = base_url


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not pulled locally."""

    def __init__(self, model_name: str):
        super().__init__(
            f"Model '{model_name}' is not available in Ollama. "
            f"Pull it first with: `ollama pull {model_name}`, "
            "or change SMALL_MODEL/LARGE_MODEL in your .env to a model you have."
        )
        self.model_name = model_name


class OllamaGenerationError(OllamaError):
    """Raised for any other failure during generation (timeout, bad response, etc.)."""
