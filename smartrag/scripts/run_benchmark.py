"""
Benchmark: baseline (naive RAG, no caching, always LARGE_MODEL) vs optimized
(SmartRAG: semantic cache + complexity routing).

Every number in the output comes from real local execution against your
running Ollama/Qdrant instance — nothing here is a fabricated or assumed
result. Run this AFTER ingesting at least one document (see
`documents/upload` or the dashboard's Documents tab), otherwise every query
will report "not available in the uploaded documents" for both arms.

Usage:
    python scripts/run_benchmark.py --queries path/to/queries.txt
    python scripts/run_benchmark.py   # uses a small built-in demo query set
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.analytics.cost import estimate_cost_usd  # noqa: E402
from backend.analytics.metrics import latency_summary  # noqa: E402
from backend.cache.semantic_cache import SemanticCache  # noqa: E402
from backend.core.config import get_settings  # noqa: E402
from backend.core.logging import get_logger  # noqa: E402
from backend.embeddings.embedder import Embedder  # noqa: E402
from backend.llm.exceptions import OllamaError  # noqa: E402
from backend.llm.ollama_client import OllamaClient  # noqa: E402
from backend.models.llm import ModelRole  # noqa: E402
from backend.rag.pipeline import RAGPipeline  # noqa: E402
from backend.rag.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from backend.retrieval.qdrant_store import QdrantStore, QdrantUnavailableError  # noqa: E402
from backend.routing.query_router import QueryRouter  # noqa: E402

logger = get_logger(__name__)

DEMO_QUERIES = [
    "What is Retrieval Augmented Generation?",
    "Can you explain what RAG means?",
    "What is semantic caching?",
    "How does semantic caching decide on a cache hit?",
    "Compare dense retrieval, sparse retrieval and hybrid retrieval, and explain their advantages and disadvantages.",
]


def load_queries(path: str | None) -> list[str]:
    if path is None:
        return DEMO_QUERIES
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def run_baseline(queries: list[str], embedder: Embedder, store: QdrantStore, llm: OllamaClient, settings) -> dict:
    """Naive RAG: no cache, always LARGE_MODEL, every query hits the LLM."""
    latencies = []
    total_tokens_all = 0
    llm_calls = 0
    errors = 0

    for q in queries:
        start = time.perf_counter()
        try:
            emb = embedder.embed(q)
            chunks = store.search(emb.vector, top_k=settings.top_k)
            if chunks:
                prompt = build_user_prompt(q, chunks)
                llm.ensure_model_available(ModelRole.LARGE)
                result = llm.generate(prompt, role=ModelRole.LARGE, system=SYSTEM_PROMPT)
                llm_calls += 1
                total_tokens_all += result.total_tokens or 0
        except (QdrantUnavailableError, OllamaError) as exc:
            errors += 1
            logger.warning(f"Baseline query failed: {exc}")
            continue
        latencies.append((time.perf_counter() - start) * 1000)

    cost = estimate_cost_usd(total_tokens_all, ModelRole.LARGE, settings)
    return {
        "total_queries": len(queries),
        "llm_calls": llm_calls,
        "errors": errors,
        "latency": latency_summary(latencies),
        "total_tokens": total_tokens_all,
        "estimated_cost_usd": cost,
    }


def run_optimized(queries: list[str], pipeline: RAGPipeline) -> dict:
    latencies = []
    cache_hits = 0
    llm_calls = 0
    errors = 0
    total_cost = 0.0

    for q in queries:
        try:
            result = pipeline.answer(q)
        except Exception as exc:  # noqa: BLE001
            errors += 1
            logger.warning(f"Optimized query failed: {exc}")
            continue
        latencies.append(result.latency.total_ms)
        if result.cache_hit:
            cache_hits += 1
        else:
            llm_calls += 1
        total_cost += result.estimated_cost_usd

    return {
        "total_queries": len(queries),
        "cache_hits": cache_hits,
        "cache_misses": len(queries) - cache_hits - errors,
        "llm_calls": llm_calls,
        "llm_calls_avoided": cache_hits,
        "errors": errors,
        "latency": latency_summary(latencies),
        "estimated_cost_usd": round(total_cost, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark baseline vs optimized SmartRAG.")
    parser.add_argument("--queries", type=str, default=None, help="Path to a text file, one query per line.")
    parser.add_argument("--output", type=str, default="benchmark_results/results.json")
    args = parser.parse_args()

    settings = get_settings()
    queries = load_queries(args.queries)

    print(f"Running benchmark with {len(queries)} queries against your local stack...")
    print("This calls your real Ollama/Qdrant/Redis instances — make sure they're running.\n")

    embedder = Embedder(settings)
    llm = OllamaClient(settings)

    if not llm.is_available():
        print(f"ERROR: Ollama is not reachable at {settings.ollama_base_url}. Start it and retry.")
        sys.exit(1)

    try:
        store = QdrantStore.from_settings(settings)
    except QdrantUnavailableError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if store.count() == 0:
        print(
            "WARNING: the knowledge base is empty. Ingest at least one document first "
            "(upload via the dashboard or POST /documents/upload) for meaningful results.\n"
        )

    print("=== Running BASELINE (no cache, always LARGE_MODEL) ===")
    baseline_results = run_baseline(queries, embedder, store, llm, settings)
    print(json.dumps(baseline_results, indent=2))

    print("\n=== Running OPTIMIZED (semantic cache + complexity routing) ===")
    cache = SemanticCache.from_settings(settings)
    cache.clear()  # start from a clean cache so results are reproducible per run
    router = QueryRouter()
    pipeline = RAGPipeline(embedder, cache, router, store, llm, analytics=None, settings=settings)
    optimized_results = run_optimized(queries, pipeline)
    print(json.dumps(optimized_results, indent=2))

    summary = {
        "queries": queries,
        "baseline": baseline_results,
        "optimized": optimized_results,
    }
    if baseline_results["latency"]["avg"] and optimized_results["latency"]["avg"]:
        summary["latency_reduction_pct"] = round(
            100 * (1 - optimized_results["latency"]["avg"] / baseline_results["latency"]["avg"]), 2
        )
    if baseline_results["estimated_cost_usd"] > 0:
        summary["cost_reduction_pct"] = round(
            100
            * (1 - optimized_results["estimated_cost_usd"] / baseline_results["estimated_cost_usd"]),
            2,
        )

    print("\n=== SUMMARY (from this run only — not a general claim) ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "queries"}, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
