"""FastAPI application entrypoint. Run via `python run.py` or
`uvicorn backend.api.app:app --reload`."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import analytics, cache, health, ingest, query
from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="SmartRAG",
    description="Semantic Caching & Cost-Aware LLM Routing — a locally-run RAG system.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local single-user tool; fine to be permissive
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(cache.router)
app.include_router(analytics.router)


@app.on_event("startup")
def on_startup() -> None:
    settings = get_settings()
    logger.info(f"SmartRAG starting | env={settings.app_env}")
    logger.info(f"  small_model={settings.small_model} large_model={settings.large_model}")
    logger.info(f"  ollama={settings.ollama_base_url} qdrant={settings.qdrant_url} redis={settings.redis_url}")
    logger.info(
        "Note: Qdrant/Redis/Ollama connectivity is checked lazily per-request; "
        "use GET /health to verify all three are reachable."
    )
