"""Pydantic contracts shared by every service.

The same models are used for Kafka payloads, Elasticsearch documents, Neo4j
projection and FastAPI responses, which keeps the six dimensions consistent.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SentimentLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


def label_from_score(score: float, neutral_band: float = 0.15) -> SentimentLabel:
    if score <= -neutral_band:
        return SentimentLabel.NEGATIVE
    if score >= neutral_band:
        return SentimentLabel.POSITIVE
    return SentimentLabel.NEUTRAL


class RawComment(BaseModel):
    """Message consumed from the Kafka ``raw_comments`` topic."""

    model_config = ConfigDict(extra="allow")

    comment_id: str
    text: str = Field(min_length=1)
    source: str = "mock"
    entity_hint: str | None = None
    author_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class AspectSentiment(BaseModel):
    """A single ABSA quadruple: (entity, aspect, opinion term, polarity)."""

    entity: str
    entity_type: str = "unknown"
    aspect_category: str
    aspect_label: str
    opinion_term: str
    sentiment: float = Field(ge=-1.0, le=1.0)
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    evidence: str = ""

    def quadruple(self) -> tuple[str, str, str, str]:
        return (self.entity, self.aspect_category, self.opinion_term, self.label.value)


class AgentVerdict(BaseModel):
    agent: Literal["vader", "finbert", "llm"]
    score: float = Field(ge=-1.0, le=1.0)
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float = 0.0
    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriAgentDecision(BaseModel):
    """Auditable output of the SDI router + committee arbitration."""

    vader_score: float
    finbert_score: float
    llm_score: float | None = None
    sdi: float = Field(ge=0.0, le=1.0)
    threshold: float
    routed_to_llm: bool
    selected_agent: Literal["vader", "finbert", "consensus", "llm"]
    final_score: float = Field(ge=-1.0, le=1.0)
    final_label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    estimated_cost_usd: float = 0.0
    avoided_cost_usd: float = 0.0
    latency_ms: float = 0.0
    agents: list[AgentVerdict] = Field(default_factory=list)


class EventRecord(BaseModel):
    """Event node projection used by Neo4j and the API."""

    event_id: str
    comment_id: str
    label: str
    event_type: str = "opinion_event"
    timestamp: datetime
    sentiment: float
    sentiment_label: SentimentLabel
    negative: bool
    aspect_category: str | None = None
    entities: list[str] = Field(default_factory=list)
    summary: str = ""
    source: str = "mock"

class ProcessedComment(BaseModel):
    """Enriched document written to Elasticsearch and Neo4j."""

    comment_id: str
    text: str
    source: str
    author_id: str | None = None
    created_at: datetime
    processed_at: datetime = Field(default_factory=utcnow)
    language: str = "zh"
    entities: list[str] = Field(default_factory=list)
    aspects: list[AspectSentiment] = Field(default_factory=list)
    sentiment: float = Field(ge=-1.0, le=1.0)
    sentiment_label: SentimentLabel
    negative: bool
    event: EventRecord | None = None
    tri_agent: TriAgentDecision
    absa_backend: str = "lexicon"
    pipeline_version: str = "2.0.0"

    @property
    def routing_audit(self) -> "RoutingAuditRecord":
        return RoutingAuditRecord(
            comment_id=self.comment_id,
            created_at=self.created_at,
            sdi=self.tri_agent.sdi,
            threshold=self.tri_agent.threshold,
            routed_to_llm=self.tri_agent.routed_to_llm,
            selected_agent=self.tri_agent.selected_agent,
            vader_score=self.tri_agent.vader_score,
            finbert_score=self.tri_agent.finbert_score,
            llm_score=self.tri_agent.llm_score,
            final_score=self.tri_agent.final_score,
            estimated_cost_usd=self.tri_agent.estimated_cost_usd,
            avoided_cost_usd=self.tri_agent.avoided_cost_usd,
            rationale=self.tri_agent.rationale,
        )


class RoutingAuditRecord(BaseModel):
    comment_id: str
    created_at: datetime
    sdi: float
    threshold: float
    routed_to_llm: bool
    selected_agent: Literal["vader", "finbert", "consensus", "llm"]
    vader_score: float
    finbert_score: float
    llm_score: float | None = None
    final_score: float
    estimated_cost_usd: float
    avoided_cost_usd: float
    rationale: str = ""


class AlertRecord(BaseModel):
    alert_id: str
    entity: str
    window_seconds: int
    sample_size: int
    negative_ratio: float
    mean_sentiment: float
    threshold: float
    triggered_at: datetime = Field(default_factory=utcnow)
    message: str
    severity: Literal["warning", "critical"] = "warning"

    @classmethod
    def build_id(cls, entity: str, window_seconds: int, bucket_ts: int) -> str:
        raw = f"{entity}:{window_seconds}:{bucket_ts}".encode()
        return hashlib.sha1(raw).hexdigest()[:16]


class GraphEvent(BaseModel):
    event_id: str
    label: str
    timestamp: datetime
    sentiment: float
    sentiment_label: SentimentLabel
    comment_id: str
    hops: int = 1


class MultiHopResult(BaseModel):
    query_entity: str
    hops: int
    negative_only: bool
    events: list[GraphEvent] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
