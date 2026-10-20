"""FastAPI query layer — dimension 6.

Endpoints
---------
GET  /health
GET  /entities/{entity}/sentiment-trend
GET  /entities/{entity}/events
GET  /graph/multihop?query=提现&hops=2&negative_only=true
GET  /routing/stats
GET  /alerts
POST /analyze           (on-demand ABSA + Tri-Agent, useful for debugging)
GET  /graph/stats
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from config.settings import Settings, settings as default_settings

from .absa_engine import ABSAEngine
from .graph_builder import GraphBuilder, build_graph_store
from .schemas import MultiHopResult
from .storage import Repository, build_repository, summarize_trend
from .tri_agent import TriAgentRouter

logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    entity_hint: str | None = None


class AnalyzeResponse(BaseModel):
    text: str
    quadruples: list[dict[str, Any]]
    tri_agent: dict[str, Any]
    backend: str


def create_app(
    config: Settings | None = None,
    *,
    repository: Repository | None = None,
    graph: GraphBuilder | None = None,
    absa_engine: ABSAEngine | None = None,
    router: TriAgentRouter | None = None,
) -> FastAPI:
    settings = config or default_settings
    repo = repository or build_repository(settings)
    graph_builder = graph
    absa = absa_engine or ABSAEngine(settings)
    tri_router = router or TriAgentRouter(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal graph_builder
        if graph_builder is None:
            graph_builder = GraphBuilder(build_graph_store(settings))
        app.state.repository = repo
        app.state.graph = graph_builder
        app.state.absa = absa
        app.state.router = tri_router
        logger.info("API ready, repo=%s", repo.health().get("backend"))
        yield
        if graph_builder is not None:
            graph_builder.close()

    app = FastAPI(
        title="Opinion & Event Graph System API",
        version="2.0.0",
        description=(
            "ABSA + Tri-Agent SDI routing + Elasticsearch + Neo4j GraphRAG "
            "for real-time crypto/fintech opinion monitoring."
        ),
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "repository": repo.health(),
            "graph": (graph_builder.stats() if graph_builder else {}),
            "llm_enabled": tri_router.llm.available,
            "sdi_threshold": settings.sdi_threshold,
        }

    @app.get("/entities/{entity}/sentiment-trend")
    def sentiment_trend(
        entity: str,
        interval: str = Query("1h", pattern=r"^[1-9][0-9]*(m|h|d)$"),
        hours: int = Query(168, ge=1, le=24 * 365),
    ) -> dict[str, Any]:
        trend = repo.entity_sentiment_trend(entity, interval=interval, hours=hours)
        if not trend:
            raise HTTPException(status_code=404, detail=f"No data for entity {entity}")
        return {
            "entity": entity,
            "interval": interval,
            "hours": hours,
            "summary": summarize_trend(trend),
            "series": trend,
        }

    @app.get("/entities/{entity}/events")
    def entity_events(
        entity: str,
        limit: int = Query(50, ge=1, le=500),
        negative_only: bool = False,
    ) -> dict[str, Any]:
        events = repo.entity_events(entity, limit=limit, negative_only=negative_only)
        return {"entity": entity, "count": len(events), "events": events}

    @app.get("/graph/multihop", response_model=MultiHopResult)
    def multihop(
        query: str = Query(..., min_length=1, max_length=64, description="实体名/方面词，如 提现"),
        hops: int = Query(2, ge=1, le=4),
        negative_only: bool = True,
        limit: int = Query(50, ge=1, le=500),
    ) -> MultiHopResult:
        if graph_builder is None:
            raise HTTPException(status_code=503, detail="Graph store is not initialised")
        return graph_builder.multihop(query, hops=hops, negative_only=negative_only, limit=limit)

    @app.get("/graph/stats")
    def graph_stats() -> dict[str, Any]:
        if graph_builder is None:
            raise HTTPException(status_code=503, detail="Graph store is not initialised")
        return graph_builder.stats()

    @app.get("/routing/stats")
    def routing_stats() -> dict[str, Any]:
        return repo.routing_stats()

    @app.get("/alerts")
    def alerts(
        entity: str | None = None,
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = repo.list_alerts(entity=entity, limit=limit)
        return {"count": len(rows), "alerts": rows}

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(payload: AnalyzeRequest) -> AnalyzeResponse:
        absa_result = absa.analyze(payload.text, payload.entity_hint)
        decision = tri_router.analyze(payload.text, absa=absa_result)
        return AnalyzeResponse(
            text=payload.text,
            quadruples=absa_result.structured,
            tri_agent=decision.model_dump(mode="json"),
            backend=absa_result.backend,
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.api_server:app",
        host=default_settings.api_host,
        port=default_settings.api_port,
        reload=False,
    )
