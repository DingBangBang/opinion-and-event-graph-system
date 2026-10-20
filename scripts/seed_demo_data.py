"""One-command offline demo: generate → process → graph → quant → figures.

This exercises exactly the same ``StreamProcessor`` used against Kafka, but with
the local JSONL repository and the in-memory graph, so it runs without Docker.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from scripts.mock_producer import generate_comments  # noqa: E402
from src.graph_builder import GraphBuilder, InMemoryGraphStore  # noqa: E402
from src.logging_conf import configure_logging  # noqa: E402
from src.quant_signal import run_analysis  # noqa: E402
from src.storage import LocalJsonlRepository, summarize_trend  # noqa: E402
from src.stream_processor import StreamProcessor  # noqa: E402


def _clean(config: Settings) -> None:
    for name in (
        "processed.jsonl",
        "routing_audit.jsonl",
        "decisions.jsonl",
        "alerts.jsonl",
        "graph_snapshot.json",
        "demo_summary.json",
        "sentiment_price.csv",
    ):
        path = config.data_dir / name
        if path.exists():
            path.unlink()


def _font_setup() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Arial Unicode MS",
        "Heiti TC",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_trends(processed: list, output: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [p.created_at for p in processed], utc=True, format="mixed"
            ),
            "entity": [p.entities[0] if p.entities else "未知实体" for p in processed],
            "sentiment": [p.sentiment for p in processed],
            "negative": [1.0 if p.negative else 0.0 for p in processed],
        }
    )
    top = frame["entity"].value_counts().head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True)
    for ax, entity in zip(axes.flat, top):
        subset = (
            frame[frame["entity"] == entity]
            .set_index("timestamp")
            .resample("1h")
            .agg(sentiment=("sentiment", "mean"), negative=("negative", "mean"))
            .dropna()
        )
        ax.plot(subset.index, subset["sentiment"], color="#d62728", label="Sentiment")
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Sentiment")
        ax2 = ax.twinx()
        ax2.bar(subset.index, subset["negative"], width=0.03, color="#ff7f0e", alpha=0.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Negative ratio")
        ax.set_title(f"{entity} — hourly sentiment & negative ratio")
        ax.grid(alpha=0.25)
    fig.suptitle("Entity-level sentiment trends (72h replay)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_routing(decisions: list, threshold: float, output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    sdis = np.array([d.sdi for d in decisions])
    agents = Counter(d.selected_agent for d in decisions)
    actual = [d.estimated_cost_usd for d in decisions]
    avoided = [d.avoided_cost_usd for d in decisions]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    axes[0].hist(sdis, bins=24, color="#1f77b4", alpha=0.8)
    axes[0].axvline(threshold, color="#d62728", linestyle="--", label=f"threshold={threshold}")
    axes[0].set_title("SDI distribution & routing threshold")
    axes[0].set_xlabel("SDI")
    axes[0].set_ylabel("comments")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    labels = list(agents)
    values = [agents[label] for label in labels]
    axes[1].bar(labels, values, color=["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"][: len(labels)])
    axes[1].set_title(f"Selected agent (LLM routing rate={agents.get('llm', 0) / len(decisions):.1%})")
    axes[1].set_ylabel("comments")
    axes[1].grid(alpha=0.25, axis="y")

    cumulative_actual = np.cumsum(actual)
    cumulative_avoided = np.cumsum(avoided)
    axes[2].plot(cumulative_avoided, label="Avoided cost (skipped calls)", color="#2ca02c")
    axes[2].plot(cumulative_actual, label="Actual LLM spend", color="#d62728")
    axes[2].set_title("Cumulative LLM cost (offline shadow price)")
    axes[2].set_xlabel("comment index")
    axes[2].set_ylabel("USD")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    fig.suptitle("Tri-Agent SDI routing economics (offline fallback: actual LLM spend = 0)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_absa(processed: list, output: Path) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = [
        {"aspect": q.aspect_label, "label": q.label.value}
        for comment in processed
        for q in comment.aspects
    ]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="aspect", columns="label", aggfunc=len, fill_value=0)
    for column in ("negative", "neutral", "positive"):
        if column not in pivot:
            pivot[column] = 0
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}
    left = pd.Series([0] * len(pivot), index=pivot.index)
    for column in ("negative", "neutral", "positive"):
        ax.barh(pivot.index, pivot[column], left=left, color=colors[column], label=column)
        left += pivot[column]
    ax.set_title("ABSA quadruples by aspect category and polarity")
    ax.set_xlabel("quadruple count")
    ax.legend()
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_graph(store: InMemoryGraphStore, output: Path) -> None:
    import matplotlib.pyplot as plt

    degree = Counter(edge["entity"] for edge in store.edges)
    entities = [name for name, _ in degree.most_common(8)]
    if not entities:
        return
    entity_set = set(entities)
    edges = [edge for edge in store.edges if edge["entity"] in entity_set]
    event_ids = list(dict.fromkeys(edge["event_id"] for edge in edges))[:24]
    event_set = set(event_ids)

    fig, ax = plt.subplots(figsize=(13, 8))
    entity_y = {name: index for index, name in enumerate(entities)}
    for name, y in entity_y.items():
        ax.scatter(0, y, s=420, color="#1f77b4", zorder=3)
        ax.text(-0.08, y, name, ha="right", va="center", fontsize=10, fontweight="bold")
    for index, event_id in enumerate(event_ids):
        event = store.events[event_id]
        y = index * (max(len(entities) - 1, 1) / max(len(event_ids) - 1, 1))
        color = "#d62728" if event.get("negative") else (
            "#2ca02c" if float(event.get("sentiment", 0)) > 0.15 else "#7f7f7f"
        )
        ax.scatter(1, y, s=120, color=color, zorder=3, alpha=0.85)
        ax.text(1.03, y, event.get("label", "")[:26], va="center", fontsize=7)
    for edge in edges:
        if edge["event_id"] not in event_set:
            continue
        event_y = event_ids.index(edge["event_id"]) * (
            max(len(entities) - 1, 1) / max(len(event_ids) - 1, 1)
        )
        ax.plot(
            [0, 1],
            [entity_y[edge["entity"]], event_y],
            color="#d62728" if edge.get("sentiment", 0) < 0 else "#2ca02c",
            alpha=0.25,
            linewidth=1,
        )
    ax.set_xlim(-0.35, 1.8)
    ax.set_title("GraphRAG preview: Entity ←(INVOLVES)– Event (red = negative)")
    ax.axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:  # pragma: no cover - demo entrypoint
    parser = argparse.ArgumentParser(description="Offline end-to-end demo seeder")
    parser.add_argument("--comments", type=int, default=520)
    parser.add_argument("--hours", type=int, default=72)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep", action="store_true", help="keep previous demo data")
    args = parser.parse_args()

    configure_logging("INFO")
    config = Settings()
    if not args.keep:
        _clean(config)
    _font_setup()

    comments = generate_comments(
        count=args.comments,
        hours=args.hours,
        seed=args.seed,
        end=datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0),
        burst=True,
    )
    print(f"→ generated {len(comments)} comments over {args.hours}h")

    repository = LocalJsonlRepository(config)
    store = InMemoryGraphStore(None)
    processor = StreamProcessor(
        config,
        repository=repository,
        graph_builder=GraphBuilder(store),
    )
    processed = []
    decisions = []
    for index, comment in enumerate(comments):
        try:
            result = processor.process_record(comment)
            processed.append(result)
            decisions.append(result.tri_agent)
        except Exception as exc:  # noqa: BLE001 - report instead of aborting demo
            processor.error_count += 1
            print(f"  ! failed {comment.comment_id}: {exc}")
        if (index + 1) % 100 == 0:
            print(f"  processed {index + 1}/{len(comments)}")
    processor.close()

    store.snapshot_path = config.data_dir / "graph_snapshot.json"
    store._persist()

    reports = config.reports_dir
    _plot_trends(processed, reports / "sentiment_trend.png")
    _plot_routing(decisions, config.sdi_threshold, reports / "routing_dashboard.png")
    _plot_absa(processed, reports / "absa_aspects.png")
    _plot_graph(store, reports / "graph_preview.png")

    quant = run_analysis(
        config.data_dir / "processed.jsonl",
        entity="币安",
        symbol="BTCUSDT",
        freq="1h",
        max_lag=12,
        live=False,
        output=reports / "sentiment_price_correlation.png",
        csv_output=config.data_dir / "sentiment_price.csv",
    )

    routing = repository.routing_stats()
    entity_counts = Counter(
        entity for comment in processed for entity in comment.entities
    )
    aspect_counts = Counter(
        q.aspect_label for comment in processed for q in comment.aspects
    )
    multihop = store.multihop("提现", hops=2, negative_only=True, limit=5)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_comments": len(comments),
        "processed_comments": len(processed),
        "errors": processor.error_count,
        "alerts": processor.alerts_emitted,
        "absa_quadruples": sum(len(comment.aspects) for comment in processed),
        "negative_share": round(
            sum(1 for comment in processed if comment.negative) / max(len(processed), 1), 4
        ),
        "avg_tri_agent_latency_ms": round(
            sum(decision.latency_ms for decision in decisions) / max(len(decisions), 1), 3
        ),
        "routing": routing,
        "graph": store.stats(),
        "top_entities": entity_counts.most_common(8),
        "top_aspects": aspect_counts.most_common(8),
        "multihop_example": {
            "query": multihop.query_entity,
            "negative_events": len(multihop.events),
            "events": [event.model_dump(mode="json") for event in multihop.events],
        },
        "quant": {k: v for k, v in quant.items() if k != "lag_table"},
        "figures": [
            "reports/sentiment_trend.png",
            "reports/routing_dashboard.png",
            "reports/absa_aspects.png",
            "reports/graph_preview.png",
            "reports/sentiment_price_correlation.png",
        ],
    }
    (config.data_dir / "demo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (reports / "demo_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== Demo summary ===")
    print(json.dumps(
        {
            "processed": summary["processed_comments"],
            "alerts": summary["alerts"],
            "absa_quadruples": summary["absa_quadruples"],
            "routing": routing,
            "graph": summary["graph"],
            "quant_best_lag": quant["best_lag_hours"],
            "quant_correlation": quant["best_lag_correlation"],
            "granger_p": quant["granger"].get("min_p_value"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":  # pragma: no cover
    main()


