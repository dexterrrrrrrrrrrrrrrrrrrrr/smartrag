from fastapi import APIRouter, Depends

from backend.api.dependencies import get_llm_client, get_qdrant_store, get_semantic_cache
from backend.api.schemas import HealthResponse
from backend.llm.ollama_client import OllamaClient
from backend.retrieval.qdrant_store import QdrantUnavailableError

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(
    llm: OllamaClient = Depends(get_llm_client),
):
    ollama_ok = llm.is_available()

    qdrant_ok = True
    try:
        get_qdrant_store().count()
    except QdrantUnavailableError:
        qdrant_ok = False

    redis_ok = get_semantic_cache().is_available()

    status = "healthy" if (ollama_ok and qdrant_ok and redis_ok) else "degraded"
    return HealthResponse(
        status=status, ollama_available=ollama_ok, qdrant_available=qdrant_ok, redis_available=redis_ok
    )
