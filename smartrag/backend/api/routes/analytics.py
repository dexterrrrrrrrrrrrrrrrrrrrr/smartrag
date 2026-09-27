from fastapi import APIRouter, Depends

from backend.analytics.repository import AnalyticsRepository
from backend.api.dependencies import get_analytics_repository

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
def overview(repo: AnalyticsRepository = Depends(get_analytics_repository)):
    return repo.overview()


@router.get("/cache")
def cache_analytics(repo: AnalyticsRepository = Depends(get_analytics_repository)):
    return repo.cache_analytics()


@router.get("/routing")
def routing_analytics(repo: AnalyticsRepository = Depends(get_analytics_repository)):
    return repo.routing_analytics()


@router.get("/performance")
def performance_analytics(repo: AnalyticsRepository = Depends(get_analytics_repository)):
    return repo.performance_analytics()


@router.delete("/clear")
def clear_analytics(repo: AnalyticsRepository = Depends(get_analytics_repository)):
    return {"rows_cleared": repo.clear()}
