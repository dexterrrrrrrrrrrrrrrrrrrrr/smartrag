"""
Full pipeline integration tests. Uses the deterministic `fake_embedder`
fixture (no network needed) plus an in-memory Qdrant store, fakeredis cache,
and a scripted fake Ollama client — exercising the exact same
`RAGPipeline` code path used in production, just with test doubles for the
two services that would otherwise require live infra.
"""
import tempfile
import time
from pathlib import Path

import pytest

from backend.analytics.db import get_engine, get_session, reset_engine_for_tests
from backend.analytics.repository import AnalyticsRepository
from backend.llm.exceptions import OllamaUnavailableError
from backend.models.documents import ChunkMetadata, DocumentChunk
from backend.models.llm import GenerationResult, ModelRole
from backend.rag.exceptions import RAGPipelineError
from backend.rag.pipeline import NO_CONTEXT_MESSAGE, RAGPipeline
from backend.retrieval.qdrant_store import QdrantUnavailableError


class ScriptedLLM:
    """A fake Ollama client that returns a fixed answer with real-looking
    (attacker-free, deterministic) token counts, or raises a scripted error."""

    def __init__(self, response_text: str = "This is a grounded answer.", tokens: int = 150, error: Exception | None = None):
        self.response_text = response_text
        self.tokens = tokens
        self.error = error
        self.calls = 0

    def ensure_model_available(self, role):
        if self.error:
            raise self.error

    def generate(self, prompt, role, system=None, temperature=0.2):
        self.calls += 1
        if self.error:
            raise self.error
        # Actually take ~self.tokens-derived time so the pipeline's own
        # wall-clock total latency measurement has something real to
        # include (a fake that reports a duration but doesn't take that
        # long would make the "total includes LLM time" assertion
        # meaningless rather than proving anything about the pipeline).
        time.sleep(0.042)
        return GenerationResult(
            text=self.response_text,
            model_name="test-model:1b",
            model_role=role,
            prompt_tokens=self.tokens - 30,
            completion_tokens=30,
            total_tokens=self.tokens,
            latency_ms=42.0,
        )


@pytest.fixture
def analytics_repo():
    tmp = tempfile.mktemp(suffix=".db")
    from backend.core.config import Settings

    settings = Settings(sqlite_db_path=tmp)
    reset_engine_for_tests()
    get_engine(settings)
    session = get_session(settings)
    yield AnalyticsRepository(session)
    session.close()
    Path(tmp).unlink(missing_ok=True)


@pytest.fixture
def seeded_store(in_memory_qdrant_store, fake_embedder):
    chunk = DocumentChunk(
        text="Retrieval Augmented Generation retrieves documents and grounds LLM answers in them.",
        metadata=ChunkMetadata(document_name="rag_notes.pdf", page_number=1, chunk_id="c1", source="rag_notes.pdf"),
    )
    in_memory_qdrant_store.upsert_chunks([chunk], [fake_embedder.embed(chunk.text).vector])
    return in_memory_qdrant_store


def test_pipeline_cache_miss_then_hit(
    fake_embedder, semantic_cache, query_router, seeded_store, analytics_repo, test_settings
):
    llm = ScriptedLLM()
    pipeline = RAGPipeline(fake_embedder, semantic_cache, query_router, seeded_store, llm, analytics_repo, test_settings)

    r1 = pipeline.answer("What is Retrieval Augmented Generation?")
    assert r1.cache_hit is False
    assert r1.answer == "This is a grounded answer."
    assert r1.sources[0].document_name == "rag_notes.pdf"
    assert llm.calls == 1

    r2 = pipeline.answer("What is Retrieval Augmented Generation?")
    assert r2.cache_hit is True
    assert r2.answer == "This is a grounded answer."
    assert llm.calls == 1  # no new LLM call on the cache hit

    overview = analytics_repo.overview()
    assert overview["total_queries"] == 2
    assert overview["cache_hits"] == 1
    assert overview["llm_calls"] == 1
    assert overview["llm_calls_avoided"] == 1


def test_missing_context_skips_llm_call(
    fake_embedder, semantic_cache, query_router, in_memory_qdrant_store, analytics_repo, test_settings
):
    """An empty knowledge base must produce an honest 'not available' answer
    without ever calling the LLM (nothing to hallucinate from)."""
    llm = ScriptedLLM()
    pipeline = RAGPipeline(
        fake_embedder, semantic_cache, query_router, in_memory_qdrant_store, llm, analytics_repo, test_settings
    )

    result = pipeline.answer("What is RAG?")

    assert result.answer == NO_CONTEXT_MESSAGE
    assert result.sources == []
    assert llm.calls == 0  # critical: no wasted/hallucination-risking LLM call


def test_cost_calculation_matches_tokens_times_price(
    fake_embedder, semantic_cache, query_router, seeded_store, analytics_repo, test_settings
):
    llm = ScriptedLLM(tokens=1000)
    test_settings.small_model_reference_cost_per_1k_tokens = 0.0002
    pipeline = RAGPipeline(fake_embedder, semantic_cache, query_router, seeded_store, llm, analytics_repo, test_settings)

    result = pipeline.answer("What is RAG?")

    assert result.total_tokens == 1000
    assert result.estimated_cost_usd == pytest.approx(0.0002)  # 1000/1000 * 0.0002


def test_latency_is_actually_measured_not_fabricated(
    fake_embedder, semantic_cache, query_router, seeded_store, analytics_repo, test_settings
):
    llm = ScriptedLLM()
    pipeline = RAGPipeline(fake_embedder, semantic_cache, query_router, seeded_store, llm, analytics_repo, test_settings)

    result = pipeline.answer("What is RAG?")

    assert result.latency.embedding_ms >= 0
    assert result.latency.cache_lookup_ms >= 0
    assert result.latency.retrieval_ms >= 0
    assert result.latency.llm_ms == 42.0  # exactly what ScriptedLLM reported
    assert result.latency.total_ms >= result.latency.llm_ms  # total includes everything else too


def test_llm_failure_raises_pipeline_error_and_logs(
    fake_embedder, semantic_cache, query_router, seeded_store, analytics_repo, test_settings
):
    llm = ScriptedLLM(error=OllamaUnavailableError("http://localhost:11434"))
    pipeline = RAGPipeline(fake_embedder, semantic_cache, query_router, seeded_store, llm, analytics_repo, test_settings)

    with pytest.raises(RAGPipelineError):
        pipeline.answer("What is RAG?")

    logs = analytics_repo.all_logs()
    assert len(logs) == 1
    assert logs[0].error is not None


def test_qdrant_failure_raises_pipeline_error(
    fake_embedder, semantic_cache, query_router, analytics_repo, test_settings
):
    class BrokenStore:
        def search(self, *args, **kwargs):
            raise QdrantUnavailableError("http://localhost:6333")

    llm = ScriptedLLM()
    pipeline = RAGPipeline(fake_embedder, semantic_cache, query_router, BrokenStore(), llm, analytics_repo, test_settings)

    with pytest.raises(RAGPipelineError):
        pipeline.answer("What is RAG?")
