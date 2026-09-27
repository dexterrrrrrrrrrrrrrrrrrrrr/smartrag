from fastapi import APIRouter, Depends

from backend.api.dependencies import get_semantic_cache
from backend.api.schemas import CacheClearResponse, CacheThresholdUpdateRequest
from backend.cache.semantic_cache import SemanticCache

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")
def cache_stats(cache: SemanticCache = Depends(get_semantic_cache)):
    return cache.stats()


@router.delete("/clear", response_model=CacheClearResponse)
def clear_cache(cache: SemanticCache = Depends(get_semantic_cache)):
    return CacheClearResponse(entries_cleared=cache.clear())


@router.put("/threshold")
def update_threshold(
    body: CacheThresholdUpdateRequest, cache: SemanticCache = Depends(get_semantic_cache)
):
    """Allows the dashboard to change the similarity threshold at runtime
    without editing .env and restarting."""
    cache.similarity_threshold = body.threshold
    return {"similarity_threshold": cache.similarity_threshold}
