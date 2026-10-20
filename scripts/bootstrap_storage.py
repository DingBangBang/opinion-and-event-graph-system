"""Bootstrap Elasticsearch indices and Neo4j constraints without processing data.

Usage:  python scripts/bootstrap_storage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402
from src.graph_builder import Neo4jGraphStore  # noqa: E402
from src.logging_conf import configure_logging  # noqa: E402
from src.storage import COMMENT_MAPPING, DECISION_MAPPING, ALERT_MAPPING  # noqa: E402


def main() -> None:
    from elasticsearch import Elasticsearch

    configure_logging("INFO")
    config = Settings()

    client = Elasticsearch([config.es_url], request_timeout=10)
    if not client.ping():
        raise SystemExit(f"Elasticsearch is not reachable at {config.es_url}")
    for index, mapping in (
        (config.es_index_comments, COMMENT_MAPPING),
        (config.es_index_decisions, DECISION_MAPPING),
        (config.es_index_alerts, ALERT_MAPPING),
    ):
        if not client.indices.exists(index=index):
            client.indices.create(
                index=index,
                mappings=mapping,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
            )
            print(f"created index {index}")
        else:
            print(f"index {index} already exists")

    store = Neo4jGraphStore(config)
    store.ensure_schema()
    store.close()
    print("Neo4j constraints/indexes are ready")


if __name__ == "__main__":
    main()
