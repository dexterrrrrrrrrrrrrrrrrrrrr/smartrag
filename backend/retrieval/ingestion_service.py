"""
Orchestrates the full ingestion pipeline:
document -> text extraction -> chunking -> embeddings -> Qdrant upsert
"""
import time
from pathlib import Path

from backend.core.logging import get_logger
from backend.embeddings.embedder import Embedder
from backend.models.documents import IngestionResult
from backend.retrieval.chunker import chunk_document
from backend.retrieval.parsers import extract_pages
from backend.retrieval.qdrant_store import QdrantStore

logger = get_logger(__name__)


class IngestionService:
    def __init__(self, embedder: Embedder, store: QdrantStore):
        self.embedder = embedder
        self.store = store

    def ingest_file(self, file_path: str | Path, document_name: str | None = None) -> IngestionResult:
        path = Path(file_path)
        document_name = document_name or path.name
        start = time.perf_counter()

        pages = extract_pages(path)
        chunks = chunk_document(document_name=document_name, pages=pages, source=str(path))

        if not chunks:
            raise ValueError(f"No chunks produced for '{document_name}' — document may be empty.")

        embeddings = [r.vector for r in self.embedder.embed_batch([c.text for c in chunks])]
        self.store.upsert_chunks(chunks, embeddings)

        latency_ms = (time.perf_counter() - start) * 1000
        total_chars = sum(len(c.text) for c in chunks)
        logger.info(
            f"Ingested '{document_name}': {len(chunks)} chunks, "
            f"{total_chars} chars, {latency_ms:.1f}ms"
        )
        return IngestionResult(
            document_name=document_name,
            num_chunks=len(chunks),
            total_characters=total_chars,
            latency_ms=latency_ms,
        )
