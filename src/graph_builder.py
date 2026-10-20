"""Knowledge graph projection and multi-hop retrieval (GraphRAG) — dimension 4.

Schema
------
``(:Entity {name, type, updated_at})``
``(:Event {event_id, label, timestamp, sentiment, negative, ...})``
``(:Event)-[:INVOLVES {timestamp, sentiment, polarity, aspect, aspect_label,
opinion, comment_id}]->(:Entity)``

``GraphBuilder`` writes the projection; ``multihop`` answers questions such as
"提现相关的负面事件时间线" by seeding on entity name *or* aspect keyword.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Protocol

from config.settings import Settings, settings as default_settings

from .event_detector import NEGATIVE_CUTOFF, build_event
from .schemas import GraphEvent, MultiHopResult, ProcessedComment, label_from_score

logger = logging.getLogger(__name__)


class GraphStore(Protocol):
    def upsert(self, comment: ProcessedComment) -> None: ...
    def multihop(
        self, query: str, hops: int = 2, negative_only: bool = False, limit: int = 50
    ) -> MultiHopResult: ...
    def stats(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class InMemoryGraphStore:
    """Deterministic graph used by tests, notebooks and the offline demo."""

    def __init__(self, snapshot_path: Path | str | None = None) -> None:
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.entities: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        if self.snapshot_path and self.snapshot_path.exists():
            self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.snapshot_path.read_text(encoding="utf-8"))  # type: ignore[union-attr]
            self.entities = payload.get("entities", {})
            self.events = payload.get("events", {})
            self.edges = payload.get("edges", [])
        except (ValueError, OSError) as exc:
            logger.warning("Could not load graph snapshot: %s", exc)

    def _persist(self) -> None:
        if not self.snapshot_path:
            return
        payload = {
            "entities": self.entities,
            "events": self.events,
            "edges": self.edges,
        }
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_path.write_text(
            json.dumps(payload, ensure_ascii=False, default=str, indent=2),
            encoding="utf-8",
        )

    def upsert(self, comment: ProcessedComment) -> None:
        event = comment.event
        if event is None:
            event = build_event(
                comment_id=comment.comment_id,
                text=comment.text,
                aspects=comment.aspects,
                entities=comment.entities,
                sentiment=comment.sentiment,
                timestamp=comment.created_at,
                source=comment.source,
            )
            comment.event = event

        for aspect in comment.aspects or []:
            self.entities.setdefault(
                aspect.entity,
                {"name": aspect.entity, "type": aspect.entity_type},
            )
            edge_key = (event.event_id, aspect.entity, aspect.aspect_category)
            if not any(
                (edge["event_id"], edge["entity"], edge["aspect"]) == edge_key
                for edge in self.edges
            ):
                self.edges.append(
                    {
                        "event_id": event.event_id,
                        "entity": aspect.entity,
                        "aspect": aspect.aspect_category,
                        "aspect_label": aspect.aspect_label,
                        "sentiment": aspect.sentiment,
                        "polarity": aspect.label.value,
                        "opinion": aspect.opinion_term,
                        "timestamp": event.timestamp.isoformat(),
                        "comment_id": comment.comment_id,
                    }
                )
        if not comment.aspects:
            for entity in event.entities:
                self.entities.setdefault(entity, {"name": entity, "type": "unknown"})
                self.edges.append(
                    {
                        "event_id": event.event_id,
                        "entity": entity,
                        "aspect": "general",
                        "aspect_label": "整体体验",
                        "sentiment": event.sentiment,
                        "polarity": event.sentiment_label.value,
                        "opinion": "整体评价",
                        "timestamp": event.timestamp.isoformat(),
                        "comment_id": comment.comment_id,
                    }
                )

        self.events[event.event_id] = event.model_dump(mode="json")
        self._persist()

    def multihop(
        self, query: str, hops: int = 2, negative_only: bool = False, limit: int = 50
    ) -> MultiHopResult:
        hops = max(1, min(4, int(hops)))
        lowered = query.lower()
        seeds: set[str] = set()
        exact_entity = query in self.entities
        if exact_entity:
            seeds.add(query)
        matched_edges: set[tuple[str, str]] = set()
        for edge in self.edges:
            event = self.events.get(edge["event_id"], {})
            if (
                edge.get("aspect") == query
                or edge.get("aspect_label") == query
                or lowered in str(event.get("label", "")).lower()
            ):
                seeds.add(edge["entity"])
                matched_edges.add((edge["event_id"], edge["entity"]))
        if not seeds:
            seeds = {
                entity
                for entity in self.entities
                if lowered in entity.lower()
            }

        adjacency: dict[str, list[dict[str, Any]]] = defaultdict(list)
        event_entities: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            adjacency[edge["entity"]].append(edge)
            event_entities[edge["event_id"]].add(edge["entity"])

        seen_entities: set[tuple[str, int]] = set()
        queue: deque[tuple[str, int, list[str]]] = deque(
            (entity, 0, [entity]) for entity in sorted(seeds)
        )
        collected: dict[str, GraphEvent] = {}
        paths: list[list[str]] = []

        while queue:
            entity, depth, path = queue.popleft()
            if depth >= hops or (entity, depth) in seen_entities:
                continue
            seen_entities.add((entity, depth))
            for edge in adjacency.get(entity, []):
                event = self.events.get(edge["event_id"])
                if not event:
                    continue
                if depth == 0 and not exact_entity:
                    if (edge["event_id"], entity) not in matched_edges:
                        continue
                event_hops = depth + 1
                negative = float(event.get("sentiment", 0.0)) < NEGATIVE_CUTOFF
                if not negative_only or negative:
                    existing = collected.get(event["event_id"])
                    if existing is None or event_hops < existing.hops:
                        collected[event["event_id"]] = GraphEvent(
                            event_id=event["event_id"],
                            label=event.get("label", ""),
                            timestamp=event["timestamp"],
                            sentiment=float(event.get("sentiment", 0.0)),
                            sentiment_label=label_from_score(
                                float(event.get("sentiment", 0.0))
                            ),
                            comment_id=event.get("comment_id", ""),
                            hops=event_hops,
                        )
                        candidate_path = path + [event.get("label", "event")]
                        if candidate_path not in paths:
                            paths.append(candidate_path)
                for other in event_entities.get(edge["event_id"], set()):
                    if other not in path:
                        queue.append(
                            (
                                other,
                                event_hops,
                                path + [event.get("label", "event"), other],
                            )
                        )

        events = sorted(
            collected.values(), key=lambda item: str(item.timestamp), reverse=True
        )[:limit]
        return MultiHopResult(
            query_entity=query,
            hops=hops,
            negative_only=negative_only,
            events=events,
            paths=paths[:limit],
        )

    def stats(self) -> dict[str, Any]:
        aspects: dict[str, int] = defaultdict(int)
        for edge in self.edges:
            aspects[edge.get("aspect", "general")] += 1
        return {
            "backend": "memory",
            "entities": len(self.entities),
            "events": len(self.events),
            "relationships": len(self.edges),
            "negative_events": sum(
                1 for event in self.events.values() if event.get("negative")
            ),
            "aspects": dict(sorted(aspects.items(), key=lambda kv: kv[1], reverse=True)),
        }

    def close(self) -> None:
        self._persist()


class Neo4jGraphStore:
    """Neo4j projection with parameterised Cypher (no string interpolation of input)."""

    def __init__(self, config: Settings | None = None) -> None:
        from neo4j import GraphDatabase

        self.settings = config or default_settings
        self.driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )
        self.driver.verify_connectivity()
        self.ensure_schema()

    def ensure_schema(self) -> None:
        statements = (
            "CREATE CONSTRAINT entity_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT event_id IF NOT EXISTS "
            "FOR (e:Event) REQUIRE e.event_id IS UNIQUE",
            "CREATE INDEX event_sentiment IF NOT EXISTS FOR (e:Event) ON (e.sentiment)",
            "CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp)",
        )
        with self.driver.session(database=self.settings.neo4j_database) as session:
            for statement in statements:
                session.run(statement).consume()

    def upsert(self, comment: ProcessedComment) -> None:
        event = comment.event
        if event is None:
            event = build_event(
                comment_id=comment.comment_id,
                text=comment.text,
                aspects=comment.aspects,
                entities=comment.entities,
                sentiment=comment.sentiment,
                timestamp=comment.created_at,
                source=comment.source,
            )
            comment.event = event

        links = [
            {
                "entity": aspect.entity,
                "entity_type": aspect.entity_type,
                "aspect": aspect.aspect_category,
                "aspect_label": aspect.aspect_label,
                "sentiment": aspect.sentiment,
                "polarity": aspect.label.value,
                "opinion": aspect.opinion_term,
            }
            for aspect in comment.aspects
        ]
        if not links:
            links = [
                {
                    "entity": entity,
                    "entity_type": "unknown",
                    "aspect": "general",
                    "aspect_label": "整体体验",
                    "sentiment": event.sentiment,
                    "polarity": event.sentiment_label.value,
                    "opinion": "整体评价",
                }
                for entity in event.entities
            ]

        query = """
        MERGE (ev:Event {event_id: $event_id})
        SET ev.label = $label,
            ev.timestamp = datetime($timestamp),
            ev.sentiment = $sentiment,
            ev.sentiment_label = $sentiment_label,
            ev.negative = $negative,
            ev.aspect_category = $aspect_category,
            ev.comment_id = $comment_id,
            ev.summary = $summary,
            ev.source = $source,
            ev.updated_at = datetime()
        WITH ev
        UNWIND $links AS link
        MERGE (en:Entity {name: link.entity})
        SET en.type = link.entity_type, en.updated_at = datetime()
        MERGE (ev)-[r:INVOLVES {aspect: link.aspect}]->(en)
        SET r.aspect_label = link.aspect_label,
            r.timestamp = datetime($timestamp),
            r.sentiment = link.sentiment,
            r.polarity = link.polarity,
            r.opinion = link.opinion,
            r.comment_id = $comment_id
        """
        params = {
            "event_id": event.event_id,
            "label": event.label,
            "timestamp": event.timestamp.isoformat(),
            "sentiment": event.sentiment,
            "sentiment_label": event.sentiment_label.value,
            "negative": event.negative,
            "aspect_category": event.aspect_category,
            "comment_id": event.comment_id,
            "summary": event.summary,
            "source": event.source,
            "links": links,
        }
        with self.driver.session(database=self.settings.neo4j_database) as session:
            session.run(query, params).consume()

    def multihop(
        self, query: str, hops: int = 2, negative_only: bool = False, limit: int = 50
    ) -> MultiHopResult:
        hops = max(1, min(4, int(hops)))
        cypher = f"""
        MATCH (seed:Entity)<-[sr:INVOLVES]-(se:Event)
        WHERE seed.name = $query
           OR sr.aspect = $query
           OR sr.aspect_label = $query
           OR toLower(se.label) CONTAINS toLower($query)
        WITH collect(DISTINCT seed) AS seeds
        UNWIND seeds AS s
        MATCH path = (s)<-[:INVOLVES*1..{hops}]-(ev:Event)
        WHERE ($negative_only = false OR ev.sentiment < $cutoff)
        WITH DISTINCT ev, path, length(path) AS hops_count
        RETURN ev.event_id AS event_id,
               ev.label AS label,
               ev.timestamp AS timestamp,
               ev.sentiment AS sentiment,
               ev.comment_id AS comment_id,
               hops_count,
               [n IN nodes(path) | coalesce(n.name, n.label)] AS path
        ORDER BY timestamp DESC
        LIMIT $limit
        """
        events: list[GraphEvent] = []
        paths: list[list[str]] = []
        with self.driver.session(database=self.settings.neo4j_database) as session:
            records = session.run(
                cypher,
                query=query,
                negative_only=negative_only,
                cutoff=NEGATIVE_CUTOFF,
                limit=limit,
            )
            for record in records:
                raw_ts = record["timestamp"]
                timestamp = (
                    raw_ts.to_native().isoformat()
                    if hasattr(raw_ts, "to_native")
                    else str(raw_ts)
                )
                events.append(
                    GraphEvent(
                        event_id=record["event_id"],
                        label=record["label"],
                        timestamp=timestamp,
                        sentiment=float(record["sentiment"]),
                        sentiment_label=label_from_score(float(record["sentiment"])),
                        comment_id=record["comment_id"],
                        hops=int(record["hops_count"]),
                    )
                )
                path = [str(item) for item in record["path"]]
                if path not in paths:
                    paths.append(path)
        return MultiHopResult(
            query_entity=query,
            hops=hops,
            negative_only=negative_only,
            events=events,
            paths=paths,
        )

    def stats(self) -> dict[str, Any]:
        with self.driver.session(database=self.settings.neo4j_database) as session:
            entities = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()
            events = session.run("MATCH (e:Event) RETURN count(e) AS n").single()
            rels = session.run("MATCH ()-[r:INVOLVES]->() RETURN count(r) AS n").single()
            negatives = session.run(
                "MATCH (e:Event) WHERE e.sentiment < $cutoff RETURN count(e) AS n",
                cutoff=NEGATIVE_CUTOFF,
            ).single()
            aspects = session.run(
                "MATCH ()-[r:INVOLVES]->() "
                "RETURN r.aspect AS aspect, count(r) AS n ORDER BY n DESC LIMIT 20"
            )
            aspect_counts = {record["aspect"]: record["n"] for record in aspects}
        return {
            "backend": "neo4j",
            "entities": entities["n"] if entities else 0,
            "events": events["n"] if events else 0,
            "relationships": rels["n"] if rels else 0,
            "negative_events": negatives["n"] if negatives else 0,
            "aspects": aspect_counts,
        }

    def close(self) -> None:
        self.driver.close()


class GraphBuilder:
    """Projection service used by the stream processor and the API."""

    def __init__(self, store: GraphStore) -> None:
        self.store = store

    def build(self, comment: ProcessedComment) -> ProcessedComment:
        if comment.event is None:
            comment.event = build_event(
                comment_id=comment.comment_id,
                text=comment.text,
                aspects=comment.aspects,
                entities=comment.entities,
                sentiment=comment.sentiment,
                timestamp=comment.created_at,
                source=comment.source,
            )
        self.store.upsert(comment)
        return comment

    def multihop(
        self, query: str, hops: int = 2, negative_only: bool = False, limit: int = 50
    ) -> MultiHopResult:
        return self.store.multihop(query, hops, negative_only, limit)

    def stats(self) -> dict[str, Any]:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()


def build_graph_store(config: Settings | None = None) -> GraphStore:
    """Prefer Neo4j; fall back to an in-memory snapshot for local demos/tests."""

    config = config or default_settings
    if config.graph_backend in {"auto", "neo4j"}:
        try:
            return Neo4jGraphStore(config)
        except Exception as exc:  # noqa: BLE001 - Neo4j may simply not be running
            if config.graph_backend == "neo4j":
                raise
            logger.warning("Neo4j unavailable (%s); using in-memory graph", exc)
    return InMemoryGraphStore(config.data_dir / "graph_snapshot.json")



