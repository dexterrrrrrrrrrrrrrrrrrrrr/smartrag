"""
Chunking logic. Uses LangChain's RecursiveCharacterTextSplitter — this is
one of the deliberate, narrow uses of LangChain (splitting text well is a
solved problem; reimplementing it wouldn't teach anything extra), while
everything downstream (embedding, storage, retrieval, caching, routing) is
implemented explicitly.
"""
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import Settings, get_settings
from backend.models.documents import ChunkMetadata, DocumentChunk


def chunk_document(
    document_name: str,
    pages: list[tuple[str, int | None]],
    source: str,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    """Split a document's extracted pages into overlapping chunks, each
    carrying full provenance metadata for later citation."""
    settings = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[DocumentChunk] = []
    for page_text, page_number in pages:
        for piece in splitter.split_text(page_text):
            if not piece.strip():
                continue
            chunks.append(
                DocumentChunk(
                    text=piece,
                    metadata=ChunkMetadata(
                        document_name=document_name,
                        page_number=page_number,
                        chunk_id=str(uuid.uuid4()),
                        source=source,
                    ),
                )
            )
    return chunks
