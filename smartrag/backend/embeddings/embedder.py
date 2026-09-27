"""
Local embedding generation via sentence-transformers.

Design goals:
- Model name and dimension always come from Settings, never hardcoded.
- The model is loaded once (lazily) and reused — loading it per-request would
  be slow and wasteful.
- Cosine similarity is implemented explicitly here (not hidden inside a
  framework call) since it's a core, interview-explainable piece of the
  semantic cache logic.
"""
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.models.embeddings import EmbeddingResult

logger = get_logger(__name__)


class Embedder:
    _model_cache: dict[str, SentenceTransformer] = {}

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.embedding_model
        self.model = self._load_model(self.model_name)

    @classmethod
    def _load_model(cls, model_name: str) -> SentenceTransformer:
        if model_name not in cls._model_cache:
            logger.info(f"Loading local embedding model '{model_name}' (first use)...")
            cls._model_cache[model_name] = SentenceTransformer(model_name)
        return cls._model_cache[model_name]

    def embed(self, text: str) -> EmbeddingResult:
        start = time.perf_counter()
        vector = self.model.encode(text, normalize_embeddings=True)
        latency_ms = (time.perf_counter() - start) * 1000
        return EmbeddingResult(
            vector=vector.tolist(),
            dimension=len(vector),
            model_name=self.model_name,
            latency_ms=latency_ms,
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        start = time.perf_counter()
        vectors = self.model.encode(texts, normalize_embeddings=True)
        total_latency_ms = (time.perf_counter() - start) * 1000
        per_item_latency_ms = total_latency_ms / max(len(texts), 1)
        return [
            EmbeddingResult(
                vector=vec.tolist(),
                dimension=len(vec),
                model_name=self.model_name,
                latency_ms=per_item_latency_ms,
            )
            for vec in vectors
        ]

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Explicit cosine similarity calculation (not delegated to a library
        call) since this is a core, must-be-explainable piece of the semantic
        cache. Vectors from `embed()` are already L2-normalized, so this
        reduces to a dot product — but we compute it generally here in case
        callers pass in non-normalized vectors from elsewhere."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
