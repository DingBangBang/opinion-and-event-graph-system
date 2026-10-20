"""Streaming pipeline: Kafka → ABSA → Tri-Agent → Elasticsearch + Neo4j.

Also implements a sliding-window negative-sentiment spike detector (dimension 3
optional requirement). The same class is reused by the offline demo seeder, so
production and demo paths share one code path.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config.settings import Settings, settings as default_settings

from .absa_engine import ABSAEngine
from .event_detector import NEGATIVE_CUTOFF
from .graph_builder import GraphBuilder, build_graph_store
from .schemas import AlertRecord, ProcessedComment, RawComment
from .storage import Repository, build_repository
from .tri_agent import RoutingAuditLogger, TriAgentRouter

logger = logging.getLogger(__name__)


class NegativeSpikeDetector:
    """Sliding window per entity; fires at most once per cooldown."""

    def __init__(
        self,
        window_seconds: int = 300,
        min_samples: int = 8,
        negative_ratio: float = 0.6,
        cooldown_seconds: int = 120,
    ) -> None:
        self.window = timedelta(seconds=window_seconds)
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.threshold = negative_ratio
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self._buffers: dict[str, deque[tuple[datetime, bool, float]]] = defaultdict(deque)
        self._last_alert: dict[str, datetime] = {}

    def add(self, comment: ProcessedComment) -> list[AlertRecord]:
        alerts: list[AlertRecord] = []
        for entity in comment.entities:
            buffer = self._buffers[entity]
            buffer.append((comment.created_at, comment.negative, comment.sentiment))
            cutoff = comment.created_at - self.window
            while buffer and buffer[0][0] < cutoff:
                buffer.popleft()

            if len(buffer) < self.min_samples:
                continue
            negatives = sum(1 for _, negative, _ in buffer if negative)
            ratio = negatives / len(buffer)
            if ratio < self.threshold:
                continue

            last = self._last_alert.get(entity)
            if last and comment.created_at - last < self.cooldown:
                continue

            self._last_alert[entity] = comment.created_at
            mean_sentiment = sum(score for _, _, score in buffer) / len(buffer)
            severity = (
                "critical" if ratio >= min(0.95, self.threshold + 0.15) else "warning"
            )
            alert = AlertRecord(
                alert_id=AlertRecord.build_id(
                    entity, self.window_seconds, int(comment.created_at.timestamp())
                ),
                entity=entity,
                window_seconds=self.window_seconds,
                sample_size=len(buffer),
                negative_ratio=round(ratio, 4),
                mean_sentiment=round(mean_sentiment, 4),
                threshold=self.threshold,
                triggered_at=comment.created_at,
                severity=severity,
                message=(
                    f"「{entity}」近 {self.window_seconds // 60} 分钟负面占比 "
                    f"{ratio:.0%}（{negatives}/{len(buffer)}），均值情感 "
                    f"{mean_sentiment:+.2f}，超过阈值 {self.threshold:.0%}"
                ),
            )
            alerts.append(alert)
        return alerts


class StreamProcessor:
    def __init__(
        self,
        config: Settings | None = None,
        *,
        repository: Repository | None = None,
        graph_builder: GraphBuilder | None = None,
        absa_engine: ABSAEngine | None = None,
        router: TriAgentRouter | None = None,
    ) -> None:
        self.settings = config or default_settings
        self.repository = repository or build_repository(self.settings)
        self.graph = graph_builder or GraphBuilder(build_graph_store(self.settings))
        self.absa = absa_engine or ABSAEngine(self.settings)
        if router is not None:
            self.router = router
        else:
            audit = RoutingAuditLogger(
                self.settings.data_dir / "routing_audit.jsonl",
                store=self.repository,
            )
            self.router = TriAgentRouter(self.settings, audit_logger=audit)
        self.detector = NegativeSpikeDetector(
            window_seconds=self.settings.alert_window_seconds,
            min_samples=self.settings.alert_min_samples,
            negative_ratio=self.settings.alert_negative_ratio,
            cooldown_seconds=self.settings.alert_cooldown_seconds,
        )
        self.processed_count = 0
        self.error_count = 0
        self.alerts_emitted = 0
        self._running = True

    def process_record(self, payload: dict[str, Any] | RawComment) -> ProcessedComment:
        raw = payload if isinstance(payload, RawComment) else RawComment.model_validate(payload)
        absa_result = self.absa.analyze(raw.text, raw.entity_hint)
        decision = self.router.analyze(
            raw.text, absa=absa_result, comment_id=raw.comment_id
        )

        entities: list[str] = []
        for quadruple in absa_result.quadruples:
            if quadruple.entity not in entities:
                entities.append(quadruple.entity)
        if not entities:
            entities = [raw.entity_hint] if raw.entity_hint else ["未知实体"]

        processed = ProcessedComment(
            comment_id=raw.comment_id,
            text=raw.text,
            source=raw.source,
            author_id=raw.author_id,
            created_at=raw.created_at,
            language=absa_result.language,
            entities=entities,
            aspects=absa_result.quadruples,
            sentiment=decision.final_score,
            sentiment_label=decision.final_label,
            negative=decision.final_score < NEGATIVE_CUTOFF,
            tri_agent=decision,
            absa_backend=absa_result.backend,
        )
        processed = self.graph.build(processed)
        self.repository.index_comment(processed)

        for alert in self.detector.add(processed):
            self.repository.index_alert(alert)
            self.alerts_emitted += 1
            logger.warning("NEGATIVE SPIKE | %s", alert.message)
            print(f"🚨 [ALERT] {alert.message}", flush=True)

        self.processed_count += 1
        if self.processed_count % 50 == 0:
            logger.info(
                "Processed %d comments (%d errors, %d alerts)",
                self.processed_count,
                self.error_count,
                self.alerts_emitted,
            )
        return processed

    def replay(self, path: str | Path, limit: int | None = None) -> int:
        """Replay a JSONL file through the exact production path."""

        count = 0
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if limit is not None and count >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    self.process_record(json.loads(line))
                    count += 1
                except Exception as exc:  # noqa: BLE001 - keep replaying the batch
                    self.error_count += 1
                    logger.error("Failed to process record: %s", exc)
        return count

    def run_kafka(
        self,
        *,
        bootstrap: str | None = None,
        topic: str | None = None,
        max_messages: int | None = None,
        from_beginning: bool = False,
    ) -> int:
        from kafka import KafkaConsumer
        from kafka.errors import KafkaError

        bootstrap = bootstrap or self.settings.kafka_bootstrap_servers
        topic = topic or self.settings.kafka_topic_raw
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap.split(","),
            group_id=self.settings.kafka_consumer_group,
            auto_offset_reset=(
                "earliest" if from_beginning else self.settings.kafka_auto_offset_reset
            ),
            enable_auto_commit=False,
            value_deserializer=lambda value: json.loads(value.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        logger.info("Consuming %s from %s", topic, bootstrap)

        def _stop(*_: Any) -> None:
            logger.info("Shutdown signal received, closing consumer…")
            self._running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        count = 0
        try:
            while self._running:
                records = consumer.poll(timeout_ms=1000, max_records=50)
                if not records:
                    continue
                for messages in records.values():
                    for message in messages:
                        try:
                            self.process_record(message.value)
                            count += 1
                        except Exception as exc:  # noqa: BLE001 - poison pill isolation
                            self.error_count += 1
                            logger.error(
                                "Failed record at partition=%s offset=%s: %s",
                                message.partition,
                                message.offset,
                                exc,
                            )
                consumer.commit()
                if max_messages is not None and count >= max_messages:
                    logger.info("Reached max_messages=%s, stopping", max_messages)
                    break
        except KafkaError as exc:
            logger.error("Kafka consumer error: %s", exc)
            raise
        finally:
            consumer.close()
        return count

    def close(self) -> None:
        self.graph.close()
        health = self.repository.health()
        logger.info(
            "Processor stopped: processed=%d errors=%d alerts=%d repo=%s",
            self.processed_count,
            self.error_count,
            self.alerts_emitted,
            health.get("backend"),
        )


def main() -> None:  # pragma: no cover - CLI glue
    from .logging_conf import configure_logging

    parser = argparse.ArgumentParser(description="Real-time opinion stream processor")
    parser.add_argument("--replay", type=str, help="JSONL file to replay offline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bootstrap", type=str, default=None)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--from-beginning", action="store_true")
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    config = Settings()
    configure_logging(args.log_level or config.log_level)
    processor = StreamProcessor(config)
    started = time.perf_counter()
    try:
        if args.replay:
            processed = processor.replay(args.replay, args.limit)
        else:
            processed = processor.run_kafka(
                bootstrap=args.bootstrap,
                topic=args.topic,
                max_messages=args.max_messages,
                from_beginning=args.from_beginning,
            )
    finally:
        processor.close()
    elapsed = time.perf_counter() - started
    print(
        f"✓ processed={processed} errors={processor.error_count} "
        f"alerts={processor.alerts_emitted} elapsed={elapsed:.2f}s "
        f"throughput={processed / elapsed if elapsed else 0:.1f} msg/s"
    )


if __name__ == "__main__":  # pragma: no cover
    main()


