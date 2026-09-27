"""
Semantic cache: stores (query embedding -> answer) pairs in Redis and finds
a hit by cosine similarity against the query embedding, not exact string
matching.

Design:
- Redis holds the source of truth as a hash per entry (`smartrag:cache:{id}`)
  containing the JSON-serialized CacheEntry, plus an index set
  (`smartrag:cache:index`) of all entry ids.
- Lookup does a linear scan over cached embeddings, computing cosine
  similarity against the incoming query. This is intentionally simple and
  explicit (interview-explainable) rather than delegating to a vector
  index; at the scale of a single-user local cache, this is fast enough,
  and the module docstring on `QueryRouter`/`Embedder` shows where a
  proper ANN index would slot in if this needed to scale.
- A Redis client is injected (`redis_client=`) so tests can pass in a
  `fakeredis` client without needing a running Redis server, while
  production code (`from_settings`) always talks to the real local Docker
  instance.
- If Redis is unreachable, `lookup()` degrades to "always miss" rather than
  raising, so the app can keep answering queries (just without caching)
  per the "continue functioning if Redis is unavailable" requirement.
"""
import time
import uuid

import redis

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.embeddings.embedder import Embedder
from backend.models.cache import CacheEntry, CacheLookupResult

logger = get_logger(__name__)

CACHE_KEY_PREFIX = "smartrag:cache:"
CACHE_INDEX_KEY = "smartrag:cache:index"


class SemanticCache:
    def __init__(
        self,
        redis_client: redis.Redis,
        similarity_threshold: float,
        ttl_seconds: int = 0,
    ):
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "SemanticCache":
        settings = settings or get_settings()
        client = redis.Redis.from_url(
            settings.redis_url, db=settings.redis_cache_db, decode_responses=True
        )
        return cls(client, settings.cache_similarity_threshold, settings.cache_ttl_seconds)

    def is_available(self) -> bool:
        try:
            return bool(self.redis.ping())
        except redis.RedisError:
            return False

    def lookup(self, query_embedding: list[float]) -> CacheLookupResult:
        start = time.perf_counter()
        try:
            entry_ids = self.redis.smembers(CACHE_INDEX_KEY)
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable during cache lookup, treating as MISS: {exc}")
            return CacheLookupResult(hit=False, lookup_latency_ms=(time.perf_counter() - start) * 1000)

        best_entry: CacheEntry | None = None
        best_score = -1.0
        checked = 0

        for entry_id in entry_ids:
            raw = self.redis.get(f"{CACHE_KEY_PREFIX}{entry_id}")
            if raw is None:
                # Expired via TTL but index wasn't cleaned up yet — self-heal.
                self.redis.srem(CACHE_INDEX_KEY, entry_id)
                continue
            entry = CacheEntry.model_validate_json(raw)
            checked += 1
            score = Embedder.cosine_similarity(query_embedding, entry.query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        lookup_latency_ms = (time.perf_counter() - start) * 1000

        if best_entry is not None and best_score >= self.similarity_threshold:
            logger.info(f"Cache HIT (similarity={best_score:.4f}) for query: {best_entry.original_query[:60]}")
            return CacheLookupResult(
                hit=True,
                entry=best_entry,
                similarity_score=best_score,
                lookup_latency_ms=lookup_latency_ms,
                candidates_checked=checked,
            )

        return CacheLookupResult(
            hit=False,
            similarity_score=best_score if best_entry is not None else None,
            lookup_latency_ms=lookup_latency_ms,
            candidates_checked=checked,
        )

    def store(self, entry: CacheEntry) -> None:
        try:
            key = f"{CACHE_KEY_PREFIX}{entry.entry_id}"
            payload = entry.model_dump_json()
            if self.ttl_seconds > 0:
                self.redis.set(key, payload, ex=self.ttl_seconds)
            else:
                self.redis.set(key, payload)
            self.redis.sadd(CACHE_INDEX_KEY, entry.entry_id)
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable, could not store cache entry: {exc}")

    def invalidate(self, entry_id: str) -> bool:
        try:
            deleted = self.redis.delete(f"{CACHE_KEY_PREFIX}{entry_id}")
            self.redis.srem(CACHE_INDEX_KEY, entry_id)
            return deleted > 0
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable, could not invalidate entry: {exc}")
            return False

    def clear(self) -> int:
        try:
            entry_ids = self.redis.smembers(CACHE_INDEX_KEY)
            if entry_ids:
                self.redis.delete(*[f"{CACHE_KEY_PREFIX}{eid}" for eid in entry_ids])
            self.redis.delete(CACHE_INDEX_KEY)
            return len(entry_ids)
        except redis.RedisError as exc:
            logger.warning(f"Redis unavailable, could not clear cache: {exc}")
            return 0

    def stats(self) -> dict:
        try:
            count = self.redis.scard(CACHE_INDEX_KEY)
            return {"available": True, "num_entries": count, "similarity_threshold": self.similarity_threshold}
        except redis.RedisError:
            return {"available": False, "num_entries": 0, "similarity_threshold": self.similarity_threshold}

    @staticmethod
    def new_entry_id() -> str:
        return str(uuid.uuid4())
