import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.api.dependencies import get_ingestion_service, get_qdrant_store
from backend.core.config import Settings, get_settings
from backend.models.documents import IngestionResult
from backend.retrieval.ingestion_service import IngestionService
from backend.retrieval.parsers import DocumentParseError, UnsupportedFileTypeError
from backend.retrieval.qdrant_store import QdrantUnavailableError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IngestionResult)
async def upload_document(
    file: UploadFile,
    ingestion: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
):
    upload_dir = Path(settings.sqlite_db_path).parent.parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingestion.ingest_file(dest_path, document_name=file.filename)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except QdrantUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return result


@router.get("/count")
def document_count(store=Depends(get_qdrant_store)):
    try:
        return {"chunk_count": store.count()}
    except QdrantUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/clear")
def clear_documents(store=Depends(get_qdrant_store)):
    try:
        store.clear()
        return {"cleared": True}
    except QdrantUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
