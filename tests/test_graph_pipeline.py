from __future__ import annotations

from datetime import datetime, timezone

from src.schemas import RawComment


def test_pipeline_writes_repository_and_graph(pipeline):
    processor, repository, store = pipeline
    now = datetime.now(timezone.utc)
    processed = processor.process_record(
        RawComment(
            comment_id="c-graph-1",
            text="币安提现三天还没到账，客服敷衍，太失望了。",
            entity_hint="币安",
            created_at=now,
        )
    )

    assert processed.event is not None
    assert processed.event.negative is True
    assert processed.tri_agent.final_label.value == "negative"
    assert processed.aspects

    comments = list(repository._read(repository.comments_path))
    assert len(comments) == 1
    assert comments[0]["comment_id"] == "c-graph-1"

    stats = store.stats()
    assert stats["entities"] >= 1
    assert stats["events"] == 1
    assert stats["relationships"] >= 1
    edge = store.edges[0]
    assert {"timestamp", "sentiment", "polarity", "aspect"}.issubset(edge)


def test_multihop_negative_events(pipeline):
    processor, _, store = pipeline
    now = datetime.now(timezone.utc)
    processor.process_record(
        RawComment(
            comment_id="c-graph-2",
            text="币安提现失败，资金冻结，准备投诉。",
            entity_hint="币安",
            created_at=now,
        )
    )
    processor.process_record(
        RawComment(
            comment_id="c-graph-3",
            text="OKX 提现秒到，手续费低，体验不错。",
            entity_hint="OKX",
            created_at=now,
        )
    )

    result = store.multihop("提现", hops=2, negative_only=True)
    assert result.query_entity == "提现"
    assert result.negative_only is True
    assert result.events, "withdrawal query should return negative events"
    assert all(event.sentiment < 0 for event in result.events)
    assert result.paths

    positive = store.multihop("提现", hops=2, negative_only=False)
    assert len(positive.events) >= len(result.events)


def test_multihop_by_entity(pipeline):
    processor, _, store = pipeline
    processor.process_record(
        RawComment(
            comment_id="c-graph-4",
            text="比特币行情暴跌，亏了很多，很难受。",
            entity_hint="BTC",
            created_at=datetime.now(timezone.utc),
        )
    )
    result = store.multihop("BTC", hops=2, negative_only=True)
    assert result.events
    assert result.events[0].event_id.startswith("evt_")


def test_event_id_is_idempotent(pipeline):
    processor, _, store = pipeline
    payload = RawComment(
        comment_id="c-graph-5",
        text="Bybit 提现失败，客服推诿。",
        entity_hint="Bybit",
        created_at=datetime.now(timezone.utc),
    )
    first = processor.process_record(payload)
    second = processor.process_record(payload)
    assert first.event is not None and second.event is not None
    assert first.event.event_id == second.event.event_id
    assert store.stats()["events"] == 1
