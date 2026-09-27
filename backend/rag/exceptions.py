"""Pipeline-level exception. Wraps underlying causes (Ollama down, Qdrant
down, etc.) so the API layer has one exception type to catch and translate
into a clean HTTP error, while still preserving the original message."""


class RAGPipelineError(Exception):
    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.cause = cause
