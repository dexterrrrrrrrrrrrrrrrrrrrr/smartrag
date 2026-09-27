from fastapi import APIRouter, Depends, HTTPException

from backend.analytics.repository import AnalyticsRepository
from backend.api.dependencies import get_analytics_repository, get_rag_pipeline
from backend.api.schemas import QueryRequest
from backend.models.rag import RAGAnswer
from backend.rag.exceptions import RAGPipelineError

router = APIRouter(tags=["query"])


@router.post("/query", response_model=RAGAnswer)
def query(
    request: QueryRequest,
    analytics: AnalyticsRepository = Depends(get_analytics_repository),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    pipeline = get_rag_pipeline(analytics)
    try:
        return pipeline.answer(request.query)
    except RAGPipelineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
