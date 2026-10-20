"""Event projection: turn one enriched comment into one graph Event node."""

from __future__ import annotations

import hashlib
from datetime import datetime

from .schemas import AspectSentiment, EventRecord, SentimentLabel, label_from_score

NEGATIVE_CUTOFF = -0.15


def _event_id(comment_id: str, aspect: str | None) -> str:
    raw = f"{comment_id}:{aspect or 'general'}".encode()
    return "evt_" + hashlib.sha1(raw).hexdigest()[:16]


def dominant_aspect(aspects: list[AspectSentiment]) -> AspectSentiment | None:
    if not aspects:
        return None
    concrete = [item for item in aspects if item.aspect_category != "general"]
    return max(concrete or aspects, key=lambda item: abs(item.sentiment))


def build_event(
    *,
    comment_id: str,
    text: str,
    aspects: list[AspectSentiment],
    entities: list[str],
    sentiment: float,
    timestamp: datetime,
    source: str = "mock",
) -> EventRecord:
    """Create a deterministic event id so Kafka retries stay idempotent."""

    main = dominant_aspect(aspects)
    entity_names = list(dict.fromkeys(
        [q.entity for q in aspects if q.entity != "未知实体"] + list(entities)
    ))
    primary_entity = entity_names[0] if entity_names else "未知实体"
    aspect_label = main.aspect_label if main else "整体体验"
    label = label_from_score(sentiment)
    tone = "负面" if sentiment < NEGATIVE_CUTOFF else (
        "正面" if sentiment > 0.15 else "中性"
    )
    event_label = f"{primary_entity}·{aspect_label}·{tone}事件"

    return EventRecord(
        event_id=_event_id(comment_id, main.aspect_category if main else None),
        comment_id=comment_id,
        label=event_label,
        event_type="opinion_event",
        timestamp=timestamp,
        sentiment=round(sentiment, 4),
        sentiment_label=label,
        negative=sentiment < NEGATIVE_CUTOFF,
        aspect_category=main.aspect_category if main else "general",
        entities=entity_names or [primary_entity],
        summary=text[:200],
        source=source,
    )


def event_sentiment_label(score: float) -> SentimentLabel:
    return label_from_score(score)
