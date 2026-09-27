"""Tests for the Qdrant-backed retrieval layer."""
from backend.models.documents import ChunkMetadata, DocumentChunk


def test_rag_retrieval_returns_relevant_chunk(in_memory_qdrant_store, fake_embedder):
    chunk = DocumentChunk(
        text="Retrieval Augmented Generation retrieves documents to ground LLM answers.",
        metadata=ChunkMetadata(document_name="notes.pdf", page_number=3, chunk_id="c1", source="notes.pdf"),
    )
    embedding = fake_embedder.embed(chunk.text).vector
    in_memory_qdrant_store.upsert_chunks([chunk], [embedding])

    query_embedding = fake_embedder.embed("What is Retrieval Augmented Generation?").vector
    results = in_memory_qdrant_store.search(query_embedding, top_k=1)

    assert len(results) == 1
    assert results[0].metadata.document_name == "notes.pdf"
    assert results[0].metadata.page_number == 3
    assert results[0].score > 0


def test_empty_document_collection_returns_no_chunks(in_memory_qdrant_store, fake_embedder):
    query_embedding = fake_embedder.embed("Anything at all?").vector
    results = in_memory_qdrant_store.search(query_embedding, top_k=5)
    assert results == []
    assert in_memory_qdrant_store.count() == 0


def test_retrieval_respects_top_k(in_memory_qdrant_store, fake_embedder):
    chunks = [
        DocumentChunk(
            text=f"Fact number {i} about retrieval augmented generation systems.",
            metadata=ChunkMetadata(document_name="doc.pdf", page_number=i, chunk_id=f"c{i}", source="doc.pdf"),
        )
        for i in range(10)
    ]
    embeddings = [fake_embedder.embed(c.text).vector for c in chunks]
    in_memory_qdrant_store.upsert_chunks(chunks, embeddings)

    results = in_memory_qdrant_store.search(fake_embedder.embed("retrieval augmented generation").vector, top_k=3)
    assert len(results) == 3


def test_clear_empties_the_collection(in_memory_qdrant_store, fake_embedder):
    chunk = DocumentChunk(
        text="Some content.",
        metadata=ChunkMetadata(document_name="doc.pdf", page_number=1, chunk_id="c1", source="doc.pdf"),
    )
    in_memory_qdrant_store.upsert_chunks([chunk], [fake_embedder.embed(chunk.text).vector])
    assert in_memory_qdrant_store.count() == 1

    in_memory_qdrant_store.clear()
    assert in_memory_qdrant_store.count() == 0
