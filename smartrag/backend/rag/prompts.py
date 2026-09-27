"""Prompt construction for context-grounded generation."""
from backend.retrieval.qdrant_store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the "
    "provided context from the user's documents. "
    "If the answer is not contained in the context, say clearly that the "
    "information is not available in the uploaded documents — do not guess "
    "or use outside knowledge. Be concise and accurate."
)


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No relevant context was found in the uploaded documents.)"
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        page = f", Page {chunk.metadata.page_number}" if chunk.metadata.page_number else ""
        parts.append(f"[Source {i}: {chunk.metadata.document_name}{page}]\n{chunk.text}")
    return "\n\n".join(parts)


def build_user_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = build_context_block(chunks)
    return (
        f"Context from uploaded documents:\n\n{context}\n\n"
        f"---\n\nQuestion: {query}\n\n"
        "Answer using only the context above. If the context does not contain "
        "the answer, say so explicitly."
    )
