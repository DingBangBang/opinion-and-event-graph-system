from __future__ import annotations

import pytest

from config.settings import Settings
from src.graph_builder import GraphBuilder, InMemoryGraphStore
from src.storage import LocalJsonlRepository
from src.stream_processor import StreamProcessor


@pytest.fixture()
def test_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        es_url="http://127.0.0.1:9",
        graph_backend="memory",
        absa_backend="lexicon",
        enable_finbert=False,
        sdi_threshold=0.45,
        deepseek_api_key="",
    )


@pytest.fixture()
def pipeline(test_settings):
    repository = LocalJsonlRepository(test_settings)
    store = InMemoryGraphStore(None)
    processor = StreamProcessor(
        test_settings,
        repository=repository,
        graph_builder=GraphBuilder(store),
    )
    yield processor, repository, store
    processor.close()
