"""
The SmartRAG pipeline. Orchestrates, in order:

  query -> embed -> semantic cache lookup
    HIT  -> return cached answer (no LLM call), log metrics
    MISS -> route (small/large model) -> retrieve from Qdrant
            -> build grounded prompt -> generate -> build sources
            -> estimate cost -> store in cache -> log metrics

Every latency field and every cost figure logged here comes from a real
measurement or a real token count times the configured reference price —
nothing is fabricated.
"""
import time

from backend.analytics.cost import estimate_avoided_cost_usd, estimate_cost_usd
from backend.analytics.repository import AnalyticsRepository
from backend.cache.semantic_cache import SemanticCache
from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.embeddings.embedder import Embedder
from backend.llm.exceptions import OllamaError
from backend.llm.ollama_client import OllamaClient
from backend.models.analytics import RequestLog
from backend.models.cache import CacheEntry, SourceCitation
from backend.models.rag import LatencyBreakdown, RAGAnswer
from backend.rag.exceptions import RAGPipelineError
from backend.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from backend.retrieval.qdrant_store import QdrantStore, QdrantUnavailableError
from backend.routing.query_router import QueryRouter

logger = get_logger(__name__)

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant information in the uploaded documents to "
    "answer this question."
)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        cache: SemanticCache,
        router: QueryRouter,
        store: QdrantStore,
        llm: OllamaClient,
        analytics: AnalyticsRepository | None = None,
        settings: Settings | None = None,
    ):
        self.embedder = embedder
        self.cache = cache
        self.router = router
        self.store = store
        self.llm = llm
        self.analytics = analytics
        self.settings = settings or get_settings()

    def answer(self, query: str) -> RAGAnswer:
        total_start = time.perf_counter()

        # 1. Embed query
        embed_result = self.embedder.embed(query)

        # 2. Semantic cache lookup
        cache_result = self.cache.lookup(embed_result.vector)

        if cache_result.hit and cache_result.entry is not None:
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            entry = cache_result.entry
            cost_if_no_cache = (
                estimate_avoided_cost_usd(entry.total_tokens, entry.model_used, self.settings)
                if entry.total_tokens
                else 0.0
            )

            self._log(
                RequestLog(
                    query=query,
                    cache_hit=True,
                    similarity_score=cache_result.similarity_score,
                    embedding_latency_ms=embed_result.latency_ms,
                    cache_lookup_latency_ms=cache_result.lookup_latency_ms,
                    total_latency_ms=total_latency_ms,
                    estimated_cost_usd=0.0,
                    estimated_cost_if_no_cache_usd=cost_if_no_cache,
                )
            )

            return RAGAnswer(
                query=query,
                answer=entry.answer,
                sources=entry.sources,
                cache_hit=True,
                similarity_score=cache_result.similarity_score,
                route=None,
                num_chunks_retrieved=0,
                total_tokens=entry.total_tokens,
                latency=LatencyBreakdown(
                    embedding_ms=embed_result.latency_ms,
                    cache_lookup_ms=cache_result.lookup_latency_ms,
                    total_ms=total_latency_ms,
                ),
                estimated_cost_usd=0.0,
                estimated_cost_if_no_cache_usd=cost_if_no_cache,
            )

        # 3. Cache MISS -> retrieve first (so the router can factor in how
        # much context was found), then route, then generate.
        retrieval_start = time.perf_counter()
        try:
            chunks = self.store.search(embed_result.vector, top_k=self.settings.top_k)
        except QdrantUnavailableError as exc:
            self._log_error(query, embed_result.latency_ms, cache_result.lookup_latency_ms, str(exc))
            raise RAGPipelineError(str(exc), exc) from exc
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        route = self.router.route(query, num_retrieved_chunks=len(chunks))

        if not chunks:
            # Nothing retrieved — answer honestly rather than hallucinating,
            # and skip the (pointless) LLM call entirely.
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            self._log(
                RequestLog(
                    query=query,
                    cache_hit=False,
                    model_role=route.role.value,
                    complexity_label=route.complexity_label,
                    complexity_score=route.signals.score,
                    num_chunks_retrieved=0,
                    embedding_latency_ms=embed_result.latency_ms,
                    cache_lookup_latency_ms=cache_result.lookup_latency_ms,
                    retrieval_latency_ms=retrieval_latency_ms,
                    total_latency_ms=total_latency_ms,
                    estimated_cost_usd=0.0,
                    estimated_cost_if_no_cache_usd=0.0,
                )
            )
            return RAGAnswer(
                query=query,
                answer=NO_CONTEXT_MESSAGE,
                sources=[],
                cache_hit=False,
                route=route,
                num_chunks_retrieved=0,
                latency=LatencyBreakdown(
                    embedding_ms=embed_result.latency_ms,
                    cache_lookup_ms=cache_result.lookup_latency_ms,
                    retrieval_ms=retrieval_latency_ms,
                    total_ms=total_latency_ms,
                ),
                estimated_cost_usd=0.0,
                estimated_cost_if_no_cache_usd=0.0,
            )

        # 4. Generate a grounded answer
        prompt = build_user_prompt(query, chunks)
        try:
            self.llm.ensure_model_available(route.role)
            generation = self.llm.generate(prompt, role=route.role, system=SYSTEM_PROMPT)
        except OllamaError as exc:
            self._log_error(
                query, embed_result.latency_ms, cache_result.lookup_latency_ms, str(exc),
                retrieval_latency_ms=retrieval_latency_ms,
            )
            raise RAGPipelineError(str(exc), exc) from exc

        total_latency_ms = (time.perf_counter() - total_start) * 1000
        sources = [
            SourceCitation(document_name=c.metadata.document_name, page_number=c.metadata.page_number)
            for c in chunks
        ]
        total_tokens = generation.total_tokens or 0
        cost = estimate_cost_usd(total_tokens, route.role, self.settings) if total_tokens else 0.0

        # 5. Store in semantic cache for future hits
        self.cache.store(
            CacheEntry(
                entry_id=SemanticCache.new_entry_id(),
                original_query=query,
                query_embedding=embed_result.vector,
                answer=generation.text,
                sources=sources,
                model_used=route.role,
                model_name=generation.model_name,
                total_tokens=total_tokens or None,
            )
        )

        # 6. Log metrics
        self._log(
            RequestLog(
                query=query,
                cache_hit=False,
                model_role=route.role.value,
                model_name=generation.model_name,
                complexity_label=route.complexity_label,
                complexity_score=route.signals.score,
                num_chunks_retrieved=len(chunks),
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                total_tokens=generation.total_tokens,
                embedding_latency_ms=embed_result.latency_ms,
                cache_lookup_latency_ms=cache_result.lookup_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=generation.latency_ms,
                total_latency_ms=total_latency_ms,
                estimated_cost_usd=cost,
                estimated_cost_if_no_cache_usd=cost,  # this call happened either way
            )
        )

        return RAGAnswer(
            query=query,
            answer=generation.text,
            sources=sources,
            cache_hit=False,
            route=route,
            num_chunks_retrieved=len(chunks),
            prompt_tokens=generation.prompt_tokens,
            completion_tokens=generation.completion_tokens,
            total_tokens=generation.total_tokens,
            latency=LatencyBreakdown(
                embedding_ms=embed_result.latency_ms,
                cache_lookup_ms=cache_result.lookup_latency_ms,
                retrieval_ms=retrieval_latency_ms,
                llm_ms=generation.latency_ms,
                total_ms=total_latency_ms,
            ),
            estimated_cost_usd=cost,
            estimated_cost_if_no_cache_usd=cost,
        )

    def _log(self, log: RequestLog) -> None:
        if self.analytics is not None:
            try:
                self.analytics.log_request(log)
            except Exception as exc:  # noqa: BLE001 - logging must never break the response
                logger.warning(f"Failed to write analytics log: {exc}")

    def _log_error(
        self,
        query: str,
        embedding_latency_ms: float,
        cache_lookup_latency_ms: float,
        error: str,
        retrieval_latency_ms: float = 0.0,
    ) -> None:
        self._log(
            RequestLog(
                query=query,
                cache_hit=False,
                embedding_latency_ms=embedding_latency_ms,
                cache_lookup_latency_ms=cache_lookup_latency_ms,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=embedding_latency_ms + cache_lookup_latency_ms + retrieval_latency_ms,
                error=error,
            )
        )
