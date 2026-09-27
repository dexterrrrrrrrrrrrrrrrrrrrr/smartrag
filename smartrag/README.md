# SmartRAG — Semantic Caching & Cost-Aware LLM Routing

A production-style, **100% free and local** RAG (Retrieval Augmented Generation) system
with an intelligent middleware layer that caches semantically similar queries and routes
each request to the cheapest local model capable of answering it well.

No OpenAI, Anthropic, or Gemini API. No paid vector DB. No paid Redis. Everything runs
on your machine via Ollama, Qdrant, and Redis.

---

## 1. What SmartRAG is

Most "chat with your PDF" demos re-run embedding → retrieval → generation for *every*
query, even when:
- A semantically identical question was already answered minutes ago.
- A tiny model could have handled a simple factual lookup just as well as a large one.

SmartRAG demonstrates the infrastructure patterns real LLM platforms use to cut latency
and inference cost without sacrificing answer quality:

1. **Semantic cache** — before doing any LLM work, check whether a sufficiently similar
   query was already answered (cosine similarity on query embeddings, not exact string
   match).
2. **Cost-aware routing** — for cache misses, an explainable heuristic decides whether a
   small or large local model handles the query.
3. **Grounded RAG generation** — retrieve relevant chunks from a local vector database
   and generate an answer that cites its sources, refusing to answer when the context
   doesn't contain the information.
4. **Real observability** — every latency, token count, cache outcome, and
   estimated-cost figure comes from an actual request. Nothing in the dashboard or
   benchmark output is fabricated.

## 2. Problem it solves

Naive RAG wastes both latency and (in a real deployment) money on:
- Duplicate/paraphrased questions that already have a good cached answer.
- Using an expensive model for questions a cheap model handles just as well.

SmartRAG measures both problems and shows, from real runs, how much a semantic cache
and a complexity router help.

## 3. Architecture

```
User Query
   │
   ▼
Query Embedding (sentence-transformers, local)
   │
   ▼
Semantic Cache Lookup (Redis + cosine similarity)
   │
   ├── HIT ──► Return cached answer (no LLM call)
   │
   └── MISS
        │
        ▼
   Qdrant Retrieval (top-K chunks)
        │
        ▼
   Query Complexity Router (heuristic, modular)
        │
        ├── LOW complexity ──► SMALL_MODEL (Ollama)
        └── HIGH complexity ─► LARGE_MODEL (Ollama)
        │
        ▼
   Context-grounded Generation
        │
        ▼
   Response + Sources ──► Store in Semantic Cache ──► Log Metrics (SQLite)
        │
        ▼
   Streamlit Dashboard (cache, routing, cost, latency analytics)
```

| Layer | Responsibility | Tech |
|---|---|---|
| API | HTTP interface, request orchestration | FastAPI |
| Embeddings | Turn text into vectors, locally | sentence-transformers (`all-MiniLM-L6-v2`) |
| Semantic Cache | Store/query cached (embedding → answer) pairs, cosine similarity | Redis |
| Vector Store | Store document chunk embeddings, similarity search | Qdrant |
| Router | Classify query complexity, pick a model | Custom heuristic (pluggable) |
| LLM | Generate grounded answers | Ollama (small/large model roles) |
| Analytics | Log every request's cache/routing/latency/cost outcome | SQLite (via SQLModel) |
| Dashboard | Visualize cache, routing, cost, latency | Streamlit + Plotly |

## 4. Semantic caching

Implemented in `backend/cache/semantic_cache.py`. Every answered query is stored in
Redis as `{query text, query embedding, answer, sources, model used, token count}`.
On a new query:

1. Embed the query.
2. Compute cosine similarity against every cached query embedding (explicit
   implementation in `backend/embeddings/embedder.py`, not hidden inside a library
   call).
3. If the best match's similarity ≥ `CACHE_SIMILARITY_THRESHOLD` (default `0.90`,
   adjustable live from the dashboard), return the cached answer — no LLM call.

**Safety property, verified by tests:** semantically *related* but *different-intent*
queries (e.g. "capital of France" vs "population of France") must not collide. This is
covered explicitly in `tests/test_cache.py::test_different_intent_not_incorrectly_cached`.

## 5. Cost-aware routing

Implemented in `backend/routing/query_router.py` as an explainable heuristic (not a
black box): query length, number of questions, comparison keywords, multi-step
keywords, and number of retrieved chunks each contribute a documented weight to a
complexity score. Above the threshold → `LARGE_MODEL`; below → `SMALL_MODEL`. The
`QueryRouter.route()` interface is designed so a learned classifier could replace the
heuristic later without touching any caller.

## 6. RAG pipeline

`backend/rag/pipeline.py` orchestrates the full flow. On a cache miss with no relevant
chunks retrieved, the pipeline **skips the LLM call entirely** and returns an honest
"not available in the uploaded documents" message rather than risking a hallucinated
answer.

## 7. Local LLM architecture

`backend/llm/ollama_client.py` never references a literal model name outside of
`Settings` — callers ask for `ModelRole.SMALL` / `ModelRole.LARGE` and the concrete
Ollama model name is resolved from `.env`. Ollama failures (server down, model not
pulled, timeout) are translated into specific, actionable exceptions rather than raw
HTTP errors.

## 8. Observability

Every request logs (in SQLite, via `backend/models/analytics.py`): cache hit/miss,
similarity score, model used, complexity label/score, chunks retrieved, token counts,
a full latency breakdown (embedding / cache lookup / retrieval / LLM / total), and
estimated cost. `backend/analytics/repository.py` computes all dashboard aggregates
(hit rate, avg/p50/p95/p99 latency, cost avoided) directly from these rows.

## 9. Benchmark methodology

`scripts/run_benchmark.py` runs the same query set through two arms against your real,
running Ollama/Qdrant/Redis:

- **Baseline**: no cache, every query always uses `LARGE_MODEL`.
- **Optimized**: SmartRAG's semantic cache + complexity router.

It reports real measured latency percentiles and real token-based cost for both arms,
plus the percentage differences **for that specific run** — it does not print a
canned "70% cost reduction" claim; the number in your terminal is whatever your actual
run produces given your queries, models, and hardware.

## 10. Installation

**Prerequisites** (all free): Python 3.11+, Docker + Docker Compose, Ollama
(https://ollama.com/download), ~6–10 GB disk for models, no GPU required.

```bash
# macOS / Linux (zsh/bash)
git clone <repo-url>
cd smartrag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

```powershell
# Windows (PowerShell)
git clone <repo-url>
cd smartrag
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

If your machine is resource-constrained, edit `.env` to use smaller models:
```
SMALL_MODEL=qwen2.5:1.5b
LARGE_MODEL=qwen2.5:3b
```
or even smaller: `phi3:mini`, `gemma2:2b`, `tinyllama`. Any two Ollama-pullable models
work — nothing is hardcoded beyond `.env`.

## 11. Ollama setup

```bash
# Install from https://ollama.com/download, then:
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b

# Verify it's serving:
curl http://localhost:11434/api/tags
```

## 12. Docker setup

```bash
docker compose up -d      # starts Qdrant (6333) and Redis (6379)

# Verify:
curl http://localhost:6333/collections
docker exec -it smartrag-redis redis-cli ping   # expect PONG
```

## 13. Running the application

```bash
# Terminal 1 — backend API
python run.py
# API docs at http://localhost:8000/docs

# Terminal 2 — dashboard
streamlit run dashboard/app.py
# opens at http://localhost:8501
```

Then, in the dashboard's **Documents** tab, upload a PDF/TXT/Markdown file, and ask
questions in the **Chat** tab.

## 14. Running tests

```bash
export PYTHONPATH=.               # Windows: set PYTHONPATH=.
pytest tests/ -v
```

53 of the 55 tests run fully offline (in-memory Qdrant, `fakeredis`, a deterministic
embedding stand-in, and scripted Ollama doubles) and require no running services. Two
tests in `tests/test_llm_client.py` are live-Ollama checks and auto-skip if Ollama
isn't running. `tests/test_embeddings.py` exercises the real `sentence-transformers`
model and needs one-time internet access to download it (~90MB) on first run.

Test coverage includes: exact and semantic cache hits, cache misses, threshold
behavior, the different-intent cache-safety property, simple/complex query routing,
token-based cost calculation, real (not fabricated) latency tracking, RAG retrieval,
missing-context handling, empty knowledge base, and graceful failure handling for
Redis, Qdrant, and Ollama outages.

## 15. Running benchmarks

```bash
# Ingest at least one document first (via the dashboard), then:
python scripts/run_benchmark.py
# or with your own query set:
python scripts/run_benchmark.py --queries my_queries.txt
```

Results are printed to the terminal and written to `benchmark_results/results.json`.

## 16. Example demo queries

**Semantic caching in action** (ingest a document about RAG/LLMs first):
1. Ask: *"What is Retrieval Augmented Generation?"* → expect a **cache MISS**, a
   retrieval + generation cycle, and the answer stored in cache.
2. Ask: *"Can you explain what RAG means?"* → expect a **cache HIT** with a high
   similarity score and zero new LLM inference.

**Cache safety** (different intent, similar surface form):
1. Ask: *"What is the capital of France?"*
2. Ask: *"What is the population of France?"* → must **not** return the capital's
   cached answer.

**Routing**:
1. Ask: *"What is RAG?"* → routes to `SMALL_MODEL`, labeled LOW complexity.
2. Ask: *"Compare dense retrieval, sparse retrieval and hybrid retrieval, and explain
   their advantages and disadvantages."* → routes to `LARGE_MODEL`, labeled HIGH
   complexity.

## 17. Screenshots

*(Add screenshots of the Chat tab and each Analytics tab here after running the app
locally.)*

## 18. What to mention on your resume

- Built a local, cost-aware RAG system with a semantic caching layer (Redis + cosine
  similarity) that avoided N% of LLM calls in benchmark runs, and a complexity-based
  model router that reduced average per-query cost by routing simple queries to a
  smaller model.
- Implemented full request observability (latency percentiles, token-based cost
  estimation, cache hit-rate tracking) backed by a real SQLite-logged dataset, exposed
  through a Streamlit analytics dashboard.
- Designed for graceful degradation: the system continues answering queries (without
  caching) if Redis is down, and surfaces clear, actionable errors if Qdrant or Ollama
  are unreachable, rather than crashing.
- Wrote 50+ automated tests covering cache correctness (including a deliberate
  different-intent false-positive safety test), routing heuristics, cost math, and
  failure-mode handling for every external dependency.

*(Fill in the "N%" with your own measured benchmark numbers — don't reuse this
template's numbers as if they were results, since your actual query set, documents,
and hardware will produce different real figures.)*

## 19. What to explain in an AI Engineer interview

- **Why cosine similarity on embeddings instead of exact string match for caching?**
  Because users paraphrase; exact match would miss "explain RAG" vs "what is RAG?".
  But similarity alone is dangerous — be ready to walk through the different-intent
  test case and why the threshold (0.90 default) matters.
- **Why route by heuristic instead of always using the large model?** Cost/latency
  tradeoff — most factual questions don't need a large model's reasoning capacity.
  Explain the specific signals used (length, comparison keywords, multi-step
  keywords, question count, retrieved-chunk count) and why each is a reasonable
  proxy for complexity.
- **How do you know the cost/latency numbers are real?** Every number traces back to
  an actual `RequestLog` row: real token counts from Ollama's response, real
  `time.perf_counter()` measurements around each pipeline stage, and a *hypothetical*
  reference price applied only for the cost estimate — clearly distinguished from an
  actual bill, because none exists here.
- **What happens when Redis/Qdrant/Ollama go down?** Redis down → cache always misses,
  app keeps answering (just uncached). Qdrant down → a typed `QdrantUnavailableError`
  surfaces as an HTTP 503 with an actionable message. Ollama down → same pattern via
  `OllamaUnavailableError`/`OllamaModelNotFoundError`.
- **Where would this need to change to scale beyond one user?** The semantic cache
  does a linear scan over all cached embeddings — fine for a single-user local cache,
  but would need a proper ANN index (e.g. Qdrant itself, or Redis's vector search) at
  scale. The router's heuristic could be replaced by a learned classifier trained on
  logged `(query, complexity_label)` outcomes.

## 20. Known limitations

- The semantic cache does a linear cosine-similarity scan over all entries in Redis —
  intentionally simple and explainable, but wouldn't scale to a very large cache
  (thousands+ of entries) without a proper vector index.
- The complexity router is a hand-tuned heuristic, not a learned model — it's
  explainable and easy to reason about, but its weights were chosen to match the two
  worked examples in the original spec, not validated against a large labeled dataset.
- PDF text extraction (`pypdf`) does not perform OCR — scanned/image-only PDFs will
  fail to ingest with a clear error rather than silently producing empty chunks.
- The cost figures are explicitly **hypothetical/reference** numbers based on a
  configurable price table — they estimate what an equivalent cloud API call *would
  have* cost, not any real expenditure.
- Single-writer SQLite is used for analytics; fine for local/single-user use, would
  need Postgres (already supported by the modular `backend/analytics/db.py` layer via
  a connection-string change) for concurrent multi-user access.

## 21. Future improvements

- Learned query router (replace the heuristic with a classifier trained on logged
  outcomes)
- Adaptive cache similarity threshold (auto-tune based on observed false-hit rate)
- Hybrid search: BM25 + vector search
- Reranking of retrieved chunks before generation
- LangGraph-based orchestration for more complex multi-step agentic flows
- OpenTelemetry tracing across the pipeline
- PostgreSQL backend for analytics (swap-in via `backend/analytics/db.py`)
- A proper ANN index for the semantic cache instead of a linear scan
- Automatic RAG evaluation (faithfulness/relevance scoring)
- Multi-model routing beyond two tiers (e.g. tiny/small/medium/large)
- Per-entry semantic cache invalidation when source documents are updated
- User-specific and privacy-aware caching for multi-tenant deployments

---

## Project structure

```
smartrag/
├── backend/
│   ├── api/            # FastAPI app, routes, request/response schemas, DI
│   ├── core/            # Settings (.env), structured logging
│   ├── rag/             # Pipeline orchestration, prompt construction
│   ├── cache/           # Redis-backed semantic cache
│   ├── routing/         # Cost-aware complexity router
│   ├── llm/              # Ollama client wrapper + typed exceptions
│   ├── embeddings/      # sentence-transformers wrapper + cosine similarity
│   ├── retrieval/       # Parsers, chunker, Qdrant store, ingestion service
│   ├── analytics/       # SQLite/SQLModel logging, cost & latency math, repository
│   └── models/          # Pydantic/SQLModel schemas shared across layers
├── dashboard/            # Streamlit app + its own thin HTTP API client
├── tests/                # 55 automated tests (pytest)
├── scripts/
│   └── run_benchmark.py  # Baseline vs optimized benchmark
├── data/
│   ├── uploads/          # Uploaded source documents (gitignored)
│   └── processed/        # (reserved for future intermediate artifacts)
├── docker-compose.yml    # Qdrant + Redis
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── run.py                # FastAPI entrypoint
```
