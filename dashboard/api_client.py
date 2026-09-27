"""Thin HTTP client the dashboard uses to talk to the FastAPI backend.
Kept separate from `backend/` so the dashboard has zero import-time
dependency on backend internals — it only ever talks over HTTP, exactly
like any other client would."""
import httpx


class ApiClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str, timeout: float = 10.0) -> dict:
        resp = httpx.get(f"{self.base_url}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, timeout: float = 10.0) -> dict:
        resp = httpx.delete(f"{self.base_url}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, json: dict, timeout: float = 10.0) -> dict:
        resp = httpx.put(f"{self.base_url}{path}", json=json, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> dict:
        return self._get("/health")

    def query(self, text: str, timeout: float = 120.0) -> dict:
        resp = httpx.post(f"{self.base_url}/query", json={"query": text}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def upload_document(self, filename: str, content: bytes, timeout: float = 300.0) -> dict:
        resp = httpx.post(
            f"{self.base_url}/documents/upload",
            files={"file": (filename, content)},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def document_count(self) -> dict:
        return self._get("/documents/count")

    def clear_documents(self) -> dict:
        return self._delete("/documents/clear")

    def cache_stats(self) -> dict:
        return self._get("/cache/stats")

    def clear_cache(self) -> dict:
        return self._delete("/cache/clear")

    def set_cache_threshold(self, threshold: float) -> dict:
        return self._put("/cache/threshold", {"threshold": threshold})

    def analytics_overview(self) -> dict:
        return self._get("/analytics/overview")

    def analytics_cache(self) -> dict:
        return self._get("/analytics/cache")

    def analytics_routing(self) -> dict:
        return self._get("/analytics/routing")

    def analytics_performance(self) -> dict:
        return self._get("/analytics/performance")

    def clear_analytics(self) -> dict:
        return self._delete("/analytics/clear")
