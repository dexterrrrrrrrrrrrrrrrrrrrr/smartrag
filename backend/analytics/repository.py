"""
Data-access layer for request logs and the aggregate metrics the dashboard
and benchmark script need. All aggregates are computed from real rows in
SQLite — nothing here is a hardcoded or simulated figure.
"""
from sqlmodel import Session, select

from backend.analytics.metrics import latency_summary
from backend.models.analytics import RequestLog


class AnalyticsRepository:
    def __init__(self, session: Session):
        self.session = session

    def log_request(self, log: RequestLog) -> RequestLog:
        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log

    def all_logs(self) -> list[RequestLog]:
        return list(self.session.exec(select(RequestLog)).all())

    def clear(self) -> int:
        logs = self.all_logs()
        for log in logs:
            self.session.delete(log)
        self.session.commit()
        return len(logs)

    def overview(self) -> dict:
        logs = self.all_logs()
        total = len(logs)
        hits = [l for l in logs if l.cache_hit]
        misses = [l for l in logs if not l.cache_hit]
        llm_calls = len(misses)  # every miss triggers exactly one generation

        total_cost = sum(l.estimated_cost_usd for l in logs)
        total_cost_if_no_cache = sum(l.estimated_cost_if_no_cache_usd for l in logs)
        cost_avoided = total_cost_if_no_cache - total_cost

        return {
            "total_queries": total,
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "cache_hit_rate": round(len(hits) / total, 4) if total else None,
            "llm_calls": llm_calls,
            "llm_calls_avoided": len(hits),
            "latency": latency_summary([l.total_latency_ms for l in logs]),
            "estimated_cost_usd": round(total_cost, 6),
            "estimated_cost_avoided_usd": round(cost_avoided, 6),
        }

    def cache_analytics(self) -> dict:
        logs = self.all_logs()
        hits = [l for l in logs if l.cache_hit]
        misses = [l for l in logs if not l.cache_hit]
        similarities = [l.similarity_score for l in logs if l.similarity_score is not None]
        return {
            "cache_hits": len(hits),
            "cache_misses": len(misses),
            "hit_rate": round(len(hits) / len(logs), 4) if logs else None,
            "similarity_scores": similarities,
            "cache_lookup_latency": latency_summary([l.cache_lookup_latency_ms for l in logs]),
            "llm_calls_avoided": len(hits),
        }

    def routing_analytics(self) -> dict:
        logs = [l for l in self.all_logs() if l.model_role is not None]
        small = [l for l in logs if l.model_role == "small"]
        large = [l for l in logs if l.model_role == "large"]
        return {
            "small_model_requests": len(small),
            "large_model_requests": len(large),
            "cost_by_model": {
                "small": round(sum(l.estimated_cost_usd for l in small), 6),
                "large": round(sum(l.estimated_cost_usd for l in large), 6),
            },
            "latency_by_model": {
                "small": latency_summary([l.total_latency_ms for l in small]),
                "large": latency_summary([l.total_latency_ms for l in large]),
            },
            "complexity_distribution": {
                "LOW": len([l for l in logs if l.complexity_label == "LOW"]),
                "HIGH": len([l for l in logs if l.complexity_label == "HIGH"]),
            },
        }

    def performance_analytics(self) -> dict:
        logs = self.all_logs()
        return {
            "total_latency": latency_summary([l.total_latency_ms for l in logs]),
            "retrieval_latency": latency_summary(
                [l.retrieval_latency_ms for l in logs if l.retrieval_latency_ms > 0]
            ),
            "llm_latency": latency_summary([l.llm_latency_ms for l in logs if l.llm_latency_ms > 0]),
            "embedding_latency": latency_summary([l.embedding_latency_ms for l in logs]),
            "timeseries": [
                {"timestamp": l.timestamp.isoformat(), "total_latency_ms": l.total_latency_ms}
                for l in sorted(logs, key=lambda x: x.timestamp)
            ],
        }
