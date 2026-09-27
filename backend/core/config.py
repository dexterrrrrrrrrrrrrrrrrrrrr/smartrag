"""
Centralized application configuration.

Everything here is read from environment variables (via `.env`). No model
names, URLs, thresholds, or pricing values are hardcoded anywhere else in
the codebase — they are always imported from an instance of `Settings`.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    small_model: str = "qwen2.5:3b"
    large_model: str = "qwen2.5:7b"
    llm_request_timeout_seconds: float = 120.0

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "smartrag_documents"

    # --- Redis ---
    redis_url: str = "redis://localhost:6379"
    redis_cache_db: int = 0

    # --- Semantic cache ---
    cache_similarity_threshold: float = 0.90
    cache_ttl_seconds: int = 604800

    # --- Ingestion / chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5

    # --- SQLite ---
    sqlite_db_path: str = "./data/smartrag.db"

    # --- Reference pricing (hypothetical, clearly labeled everywhere used) ---
    small_model_reference_cost_per_1k_tokens: float = 0.0002
    large_model_reference_cost_per_1k_tokens: float = 0.0020

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor. Import and call this everywhere instead of
    instantiating Settings() directly, so the whole app shares one instance."""
    return Settings()
