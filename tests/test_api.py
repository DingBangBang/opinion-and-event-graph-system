from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api_server import create_app
from src.graph_builder import GraphBuilder
from src.schemas import RawComment


def _build_client(pipeline):
    processor, repository, store = pipeline
    now = datetime.now(timezone.utc)
    processor.process_record(
        RawComment(
            comment_id="api-1",
            text="币安提现三天还没到账，客服敷衍，太失望了。",
            entity_hint="币安",
            created_at=now,
        )
    )
    app = create_app(
        processor.settings,
        repository=repository,
        graph=GraphBuilder(store),
        absa_engine=processor.absa,
        router=processor.router,
    )
    return TestClient(app)


def test_health(pipeline):
    with _build_client(pipeline) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["repository"]["backend"] == "local-jsonl"


def test_sentiment_trend(pipeline):
    with _build_client(pipeline) as client:
        response = client.get("/entities/币安/sentiment-trend?interval=1h&hours=24")
        assert response.status_code == 200
        body = response.json()
        assert body["entity"] == "币安"
        assert body["series"]
        assert body["summary"]["negative_ratio"] > 0


def test_entity_events(pipeline):
    with _build_client(pipeline) as client:
        response = client.get("/entities/币安/events?negative_only=true")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 1


def test_multihop_endpoint(pipeline):
    with _build_client(pipeline) as client:
        response = client.get("/graph/multihop?query=提现&hops=2&negative_only=true")
        assert response.status_code == 200
        body = response.json()
        assert body["query_entity"] == "提现"
        assert body["events"]


def test_analyze_endpoint(pipeline):
    with _build_client(pipeline) as client:
        response = client.post(
            "/analyze", json={"text": "OKX 提现秒到，手续费低，体验不错。"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["quadruples"]
        assert body["tri_agent"]["final_score"] >= 0


def test_routing_and_alerts_endpoints(pipeline):
    with _build_client(pipeline) as client:
        stats = client.get("/routing/stats")
        assert stats.status_code == 200
        assert stats.json()["total"] >= 1
        alerts = client.get("/alerts")
        assert alerts.status_code == 200
        assert "alerts" in alerts.json()
