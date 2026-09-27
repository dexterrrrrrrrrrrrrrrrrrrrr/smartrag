"""
Qdrant vector store wrapper for document chunks.

Design goals:
- The concrete QdrantClient is injectable (`client=` param) so tests can pass
  an in-memory client (`QdrantClient(location=":memory:")`) without needing a
  running Qdrant server, while production code (via `from_settings`) always
  talks to the real local Docker instance.
- Qdrant connection failures are translated into a specific exception so the
  API layer can return a clean 503 instead of a raw stack trace.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.models.documents import ChunkMetadata, DocumentChunk

logger = get_logger(__name__)


class QdrantUnavailableError(Exception):
    def __init__(self, url: str, cause: Exception | None = None):
        super().__init__(
            f"Could not reach Qdrant at {url}. "
            "Is it running? Try: `docker compose up -d`, then retry."
        )
        self.url = url
        self.cause = cause


class RetrievedChunk:
    """Plain result object for a retrieval hit (kept simple/explicit rather
    than wrapping Qdrant's own point type everywhere downstream)."""

    def __init__(self, text: str, metadata: ChunkMetadata, score: float):
        self.text = text
        self.metadata = metadata
        self.score = score


class QdrantStore:
    def __init__(self, client: QdrantClient, collection_name: str, embedding_dimension: int):
        self.client = client
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._ensure_collection()

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "QdrantStore":
        settings = settings or get_settings()
        try:
            client = QdrantClient(url=settings.qdrant_url)
        except Exception as exc:  # noqa: BLE001
            raise QdrantUnavailableError(settings.qdrant_url, exc) from exc
        return cls(client, settings.qdrant_collection_name, settings.embedding_dimension)

    def _ensure_collection(self) -> None:
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if self.collection_name not in existing:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}'")
        except UnexpectedResponse as exc:
            raise QdrantUnavailableError(self.collection_name, exc) from exc
        except Exception as exc:  # noqa: BLE001 - connection refused etc. arrive as generic errors
            raise QdrantUnavailableError(self.collection_name, exc) from exc

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "document_name": chunk.metadata.document_name,
                    "page_number": chunk.metadata.page_number,
                    "chunk_id": chunk.metadata.chunk_id,
                    "source": chunk.metadata.source,
                    "created_at": chunk.metadata.created_at.isoformat(),
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            raise QdrantUnavailableError(self.collection_name, exc) from exc
        return len(points)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        document_name: str | None = None,
    ) -> list[RetrievedChunk]:
        query_filter = None
        if document_name:
            query_filter = Filter(
                must=[FieldCondition(key="document_name", match=MatchValue(value=document_name))]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                query_filter=query_filter,
            ).points
        except Exception as exc:  # noqa: BLE001
            raise QdrantUnavailableError(self.collection_name, exc) from exc

        retrieved: list[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            retrieved.append(
                RetrievedChunk(
                    text=payload.get("text", ""),
                    metadata=ChunkMetadata(
                        document_name=payload.get("document_name", "unknown"),
                        page_number=payload.get("page_number"),
                        chunk_id=payload.get("chunk_id", ""),
                        source=payload.get("source", ""),
                    ),
                    score=point.score,
                )
            )
        return retrieved

    def count(self) -> int:
        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception as exc:  # noqa: BLE001
            raise QdrantUnavailableError(self.collection_name, exc) from exc

    def clear(self) -> None:
        """Delete and recreate the collection — used by tests and the
        dashboard's 'clear knowledge base' action."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - fine if it didn't exist
            pass
        self._ensure_collection()
