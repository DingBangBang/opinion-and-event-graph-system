"""Persistence layer for Elasticsearch with a transparent local fallback.

The API and the stream processor share one interface:

* :class:`ElasticsearchRepository` — production path (aggregations, full-text)
* :class:`LocalJsonlRepository` — offline demo / tests (no Docker required)

Keeping both behind the same methods is what makes the repository runnable on a
laptop while still demonstrating the real-time architecture.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Protocol

from config.settings import Settings, settings as default_settings

from .schemas import AlertRecord, ProcessedComment, RoutingAuditRecord

logger = logging.getLogger(__name__)


COMMENT_MAPPING: dict[str, Any] = {
    "properties": {
        "comment_id": {"type": "keyword"},
        "text": {"type": "text", "fields": {"raw": {"type": "keyword", "ignore_above": 512}}},
        "source": {"type": "keyword"},
        "author_id": {"type": "keyword"},
        "created_at": {"type": "date"},
        "processed_at": {"type": "date"},
        "language": {"type": "keyword"},
        "entities": {"type": "keyword"},
        "sentiment": {"type": "float"},
        "sentiment_label": {"type": "keyword"},
        "negative": {"type": "boolean"},
        "absa_backend": {"type": "keyword"},
        "pipeline_version": {"type": "keyword"},
        "aspects": {
            "type": "nested",
            "properties": {
                "entity": {"type": "keyword"},
                "entity_type": {"type": "keyword"},
                "aspect_category": {"type": "keyword"},
                "aspect_label": {"type": "keyword"},
                "opinion_term": {"type": "text"},
                "sentiment": {"type": "float"},
                "label": {"type": "keyword"},
                "confidence": {"type": "float"},
            },
        },
        "event": {
            "type": "object",
            "properties": {
                "event_id": {"type": "keyword"},
                "label": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "timestamp": {"type": "date"},
                "sentiment": {"type": "float"},
                "negative": {"type": "boolean"},
                "aspect_category": {"type": "keyword"},
                "entities": {"type": "keyword"},
            },
        },
        "tri_agent": {
            "type": "object",
            "properties": {
                "sdi": {"type": "float"},
                "routed_to_llm": {"type": "boolean"},
                "selected_agent": {"type": "keyword"},
                "final_score": {"type": "float"},
                "estimated_cost_usd": {"type": "float"},
                "avoided_cost_usd": {"type": "float"},
            },
        },
    }
}

DECISION_MAPPING: dict[str, Any] = {
    "properties": {
        "comment_id": {"type": "keyword"},
        "created_at": {"type": "date"},
        "sdi": {"type": "float"},
        "threshold": {"type": "float"},
        "routed_to_llm": {"type": "boolean"},
        "selected_agent": {"type": "keyword"},
        "vader_score": {"type": "float"},
        "finbert_score": {"type": "float"},
        "llm_score": {"type": "float"},
        "final_score": {"type": "float"},
        "estimated_cost_usd": {"type": "float"},
        "avoided_cost_usd": {"type": "float"},
    }
}

ALERT_MAPPING: dict[str, Any] = {
    "properties": {
        "alert_id": {"type": "keyword"},
        "entity": {"type": "keyword"},
        "window_seconds": {"type": "integer"},
        "sample_size": {"type": "integer"},
        "negative_ratio": {"type": "float"},
        "mean_sentiment": {"type": "float"},
        "threshold": {"type": "float"},
        "triggered_at": {"type": "date"},
        "message": {"type": "text"},
        "severity": {"type": "keyword"},
    }
}


class Repository(Protocol):
    def index_comment(self, comment: ProcessedComment) -> None: ...
    def index_decision(self, record: RoutingAuditRecord) -> None: ...
    def index_alert(self, alert: AlertRecord) -> None: ...
    def entity_sentiment_trend(
        self, entity: str, interval: str = "1h", hours: int = 168
    ) -> list[dict[str, Any]]: ...
    def entity_events(
        self, entity: str, limit: int = 50, negative_only: bool = False
    ) -> list[dict[str, Any]]: ...
    def routing_stats(self) -> dict[str, Any]: ...
    def list_alerts(self, entity: str | None, limit: int = 50) -> list[dict[str, Any]]: ...
    def health(self) -> dict[str, Any]: ...


class ElasticsearchRepository:
    def __init__(self, config: Settings | None = None) -> None:
        from elasticsearch import Elasticsearch

        self.settings = config or default_settings
        auth = (
            (self.settings.es_username, self.settings.es_password)
            if self.settings.es_username
            else None
        )
        self.client = Elasticsearch(
            [self.settings.es_url],
            basic_auth=auth,
            request_timeout=10,
            retry_on_timeout=True,
            max_retries=2,
        )
        self.available = bool(self.client.ping())
        if self.available:
            self.ensure_indices()

    def ensure_indices(self) -> None:
        for index, mapping in (
            (self.settings.es_index_comments, COMMENT_MAPPING),
            (self.settings.es_index_decisions, DECISION_MAPPING),
            (self.settings.es_index_alerts, ALERT_MAPPING),
        ):
            if not self.client.indices.exists(index=index):
                self.client.indices.create(
                    index=index,
                    mappings=mapping,
                    settings={"number_of_shards": 1, "number_of_replicas": 0},
                )
                logger.info("Created Elasticsearch index %s", index)

    def index_comment(self, comment: ProcessedComment) -> None:
        self.client.index(
            index=self.settings.es_index_comments,
            id=comment.comment_id,
            document=comment.model_dump(mode="json"),
        )

    def index_decision(self, record: RoutingAuditRecord) -> None:
        self.client.index(
            index=self.settings.es_index_decisions,
            document=record.model_dump(mode="json"),
        )

    def index_alert(self, alert: AlertRecord) -> None:
        self.client.index(
            index=self.settings.es_index_alerts,
            id=alert.alert_id,
            document=alert.model_dump(mode="json"),
        )

    def entity_sentiment_trend(
        self, entity: str, interval: str = "1h", hours: int = 168
    ) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.settings.es_index_comments,
            size=0,
            query={
                "bool": {
                    "filter": [
                        {"term": {"entities": entity}},
                        {"range": {"created_at": {"gte": f"now-{hours}h"}}},
                    ]
                }
            },
            aggs={
                "timeline": {
                    "date_histogram": {
                        "field": "created_at",
                        "fixed_interval": interval,
                        "min_doc_count": 0,
                    },
                    "aggs": {
                        "avg_sentiment": {"avg": {"field": "sentiment"}},
                        "negative_hits": {"filter": {"term": {"negative": True}}},
                    },
                }
            },
        )
        buckets = response.get("aggregations", {}).get("timeline", {}).get("buckets", [])
        trend: list[dict[str, Any]] = []
        for bucket in buckets:
            count = bucket["doc_count"]
            trend.append(
                {
                    "timestamp": bucket["key_as_string"],
                    "count": count,
                    "avg_sentiment": round(
                        float(bucket["avg_sentiment"]["value"] or 0.0), 4
                    ),
                    "negative_ratio": round(
                        bucket["negative_hits"]["doc_count"] / count, 4
                    ) if count else 0.0,
                }
            )
        return trend

    def entity_events(
        self, entity: str, limit: int = 50, negative_only: bool = False
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [{"term": {"entities": entity}}]
        if negative_only:
            filters.append({"term": {"negative": True}})
        response = self.client.search(
            index=self.settings.es_index_comments,
            size=limit,
            query={"bool": {"filter": filters}},
            sort=[{"event.timestamp": {"order": "desc"}}],
            source=[
                "comment_id", "text", "entities", "sentiment",
                "event", "aspects", "created_at",
            ],
        )
        events: list[dict[str, Any]] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit["_source"]
            event = source.get("event") or {}
            events.append(
                {
                    "event_id": event.get("event_id", hit["_id"]),
                    "comment_id": source.get("comment_id"),
                    "label": event.get("label", (source.get("text") or "")[:40]),
                    "timestamp": event.get("timestamp") or source.get("created_at"),
                    "sentiment": event.get("sentiment", source.get("sentiment", 0.0)),
                    "sentiment_label": event.get("sentiment_label", "neutral"),
                    "entities": event.get("entities", source.get("entities", [])),
                    "aspect_category": event.get("aspect_category"),
                    "text": source.get("text", ""),
                }
            )
        return events

    def routing_stats(self) -> dict[str, Any]:
        response = self.client.search(
            index=self.settings.es_index_decisions,
            size=0,
            aggs={
                "agents": {"terms": {"field": "selected_agent", "size": 10}},
                "routed": {"filter": {"term": {"routed_to_llm": True}}},
                "avg_sdi": {"avg": {"field": "sdi"}},
                "cost": {"sum": {"field": "estimated_cost_usd"}},
                "avoided": {"sum": {"field": "avoided_cost_usd"}},
            },
        )
        aggs = response.get("aggregations", {})
        routed = int(aggs.get("routed", {}).get("doc_count", 0))
        all_docs = response.get("hits", {}).get("total", {})
        total_docs = int(all_docs.get("value", 0)) if isinstance(all_docs, dict) else 0
        return {
            "total": total_docs,
            "routed_to_llm": routed,
            "routing_rate": round(routed / total_docs, 4) if total_docs else 0.0,
            "avg_sdi": round(float(aggs.get("avg_sdi", {}).get("value") or 0.0), 4),
            "estimated_cost_usd": round(float(aggs.get("cost", {}).get("value") or 0.0), 8),
            "avoided_cost_usd": round(float(aggs.get("avoided", {}).get("value") or 0.0), 8),
            "by_agent": {
                bucket["key"]: bucket["doc_count"]
                for bucket in aggs.get("agents", {}).get("buckets", [])
            },
        }

    def list_alerts(
        self, entity: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"match_all": {}}
        if entity:
            query = {"term": {"entity": entity}}
        response = self.client.search(
            index=self.settings.es_index_alerts,
            size=limit,
            query=query,
            sort=[{"triggered_at": {"order": "desc"}}],
        )
        return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]

    def health(self) -> dict[str, Any]:
        try:
            count = self.client.count(index=self.settings.es_index_comments)["count"]
        except Exception:  # noqa: BLE001
            count = 0
        return {
            "backend": "elasticsearch",
            "available": self.available,
            "url": self.settings.es_url,
            "documents": int(count),
        }


class LocalJsonlRepository:
    """Offline repository so the demo/tests need no infrastructure."""

    def __init__(self, config: Settings | None = None) -> None:
        self.settings = config or default_settings
        self.settings.ensure_dirs()
        self.comments_path = self.settings.data_dir / "processed.jsonl"
        self.decisions_path = self.settings.data_dir / "decisions.jsonl"
        self.audit_path = self.settings.data_dir / "routing_audit.jsonl"
        self.alerts_path = self.settings.data_dir / "alerts.jsonl"

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _read(path: Path) -> Iterable[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def index_comment(self, comment: ProcessedComment) -> None:
        self._append(self.comments_path, comment.model_dump(mode="json"))

    def index_decision(self, record: RoutingAuditRecord) -> None:
        self._append(self.decisions_path, record.model_dump(mode="json"))

    def index_alert(self, alert: AlertRecord) -> None:
        self._append(self.alerts_path, alert.model_dump(mode="json"))

    def entity_sentiment_trend(
        self, entity: str, interval: str = "1h", hours: int = 168
    ) -> list[dict[str, Any]]:
        import pandas as pd

        rows = [
            row
            for row in self._read(self.comments_path)
            if entity in (row.get("entities") or [])
        ]
        if not rows:
            return []
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [row.get("created_at") for row in rows], utc=True, format="mixed"
                ),
                "sentiment": [float(row.get("sentiment", 0.0)) for row in rows],
                "negative": [1.0 if row.get("negative") else 0.0 for row in rows],
            }
        )
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        frame = frame[frame["timestamp"] >= cutoff]
        if frame.empty:
            return []
        rule = {"1h": "1h", "30m": "30min", "15m": "15min", "1d": "1D"}.get(
            interval, interval
        )
        grouped = (
            frame.set_index("timestamp")
            .resample(rule)
            .agg(
                count=("sentiment", "size"),
                avg_sentiment=("sentiment", "mean"),
                negative_ratio=("negative", "mean"),
            )
            .reset_index()
        )
        return [
            {
                "timestamp": row.timestamp.isoformat(),
                "count": int(row.count),
                "avg_sentiment": round(float(row.avg_sentiment), 4)
                if row.count else 0.0,
                "negative_ratio": round(float(row.negative_ratio), 4)
                if row.count else 0.0,
            }
            for row in grouped.itertuples()
        ]

    def entity_events(
        self, entity: str, limit: int = 50, negative_only: bool = False
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for row in self._read(self.comments_path):
            entities = row.get("entities") or [
                aspect.get("entity") for aspect in row.get("aspects", [])
            ]
            if entity not in entities:
                continue
            if negative_only and not row.get("negative"):
                continue
            event = row.get("event") or {}
            events.append(
                {
                    "event_id": event.get("event_id", row.get("comment_id")),
                    "comment_id": row.get("comment_id"),
                    "label": event.get("label", (row.get("text") or "")[:40]),
                    "timestamp": event.get("timestamp") or row.get("created_at"),
                    "sentiment": event.get("sentiment", row.get("sentiment", 0.0)),
                    "sentiment_label": event.get(
                        "sentiment_label", row.get("sentiment_label", "neutral")
                    ),
                    "entities": event.get("entities", entities),
                    "aspect_category": event.get("aspect_category"),
                    "text": row.get("text", ""),
                }
            )
        events.sort(key=lambda item: str(item["timestamp"]), reverse=True)
        return events[:limit]

    def routing_stats(self) -> dict[str, Any]:
        rows = list(self._read(self.audit_path))
        if not rows:
            rows = list(self._read(self.decisions_path))
        if not rows:
            return {
                "total": 0,
                "routed_to_llm": 0,
                "routing_rate": 0.0,
                "avg_sdi": 0.0,
                "estimated_cost_usd": 0.0,
                "avoided_cost_usd": 0.0,
                "by_agent": {},
            }
        agents: Counter[str] = Counter(row.get("selected_agent", "unknown") for row in rows)
        routed = sum(1 for row in rows if row.get("routed_to_llm"))
        return {
            "total": len(rows),
            "routed_to_llm": routed,
            "routing_rate": round(routed / len(rows), 4),
            "avg_sdi": round(sum(float(row.get("sdi", 0.0)) for row in rows) / len(rows), 4),
            "estimated_cost_usd": round(
                sum(float(row.get("estimated_cost_usd", 0.0)) for row in rows), 8
            ),
            "avoided_cost_usd": round(
                sum(float(row.get("avoided_cost_usd", 0.0)) for row in rows), 8
            ),
            "by_agent": dict(agents),
        }

    def list_alerts(
        self, entity: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        rows = list(self._read(self.alerts_path))
        if entity:
            rows = [row for row in rows if row.get("entity") == entity]
        rows.sort(key=lambda item: str(item.get("triggered_at", "")), reverse=True)
        return rows[:limit]

    def health(self) -> dict[str, Any]:
        return {
            "backend": "local-jsonl",
            "available": True,
            "path": str(self.comments_path),
            "documents": len(list(self._read(self.comments_path))),
        }


def build_repository(
    config: Settings | None = None, prefer_elasticsearch: bool = True
) -> Repository:
    """Return Elasticsearch when reachable, otherwise the local JSONL store."""

    config = config or default_settings
    if prefer_elasticsearch:
        try:
            repository: ElasticsearchRepository = ElasticsearchRepository(config)
            if repository.available:
                return repository
        except Exception as exc:  # noqa: BLE001 - connection issues are expected offline
            logger.warning("Elasticsearch unavailable (%s); using local JSONL store", exc)
    return LocalJsonlRepository(config)


def summarize_trend(trend: list[dict[str, Any]]) -> dict[str, Any]:
    """Small helper used by the API and reports."""

    if not trend:
        return {"points": 0, "avg_sentiment": 0.0, "negative_ratio": 0.0}
    weighted = [row for row in trend if row.get("count")]
    if not weighted:
        return {"points": len(trend), "avg_sentiment": 0.0, "negative_ratio": 0.0}
    total = sum(row["count"] for row in weighted)
    return {
        "points": len(trend),
        "avg_sentiment": round(
            sum(row["avg_sentiment"] * row["count"] for row in weighted) / total, 4
        ),
        "negative_ratio": round(
            sum(row["negative_ratio"] * row["count"] for row in weighted) / total, 4
        ),
    }


__all__ = [
    "AlertRecord",
    "ElasticsearchRepository",
    "LocalJsonlRepository",
    "Repository",
    "build_repository",
    "summarize_trend",
]


