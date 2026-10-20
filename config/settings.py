"""Centralised, environment-driven configuration.

All services (API, stream processor, quant job, scripts) read the same settings
object so that local runs and Docker Compose behave identically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration loaded from ``.env`` + process environment."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- generic ----------
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"
    timezone: str = "Asia/Shanghai"

    # ---------- Kafka ----------
    kafka_bootstrap_servers: str = "localhost:29092"
    kafka_topic_raw: str = "raw_comments"
    kafka_consumer_group: str = "opinion-stream-processor"
    kafka_auto_offset_reset: str = "latest"

    # ---------- Elasticsearch ----------
    es_url: str = "http://localhost:9200"
    es_index_comments: str = "opinion_comments"
    es_index_decisions: str = "tri_agent_decisions"
    es_index_alerts: str = "opinion_alerts"
    es_username: str = ""
    es_password: str = ""

    # ---------- Neo4j / graph ----------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "opinion-graph"
    neo4j_database: str = "neo4j"
    graph_backend: Literal["auto", "neo4j", "memory"] = "auto"

    # ---------- ABSA ----------
    absa_backend: Literal["auto", "lexicon", "hf", "deepseek"] = "auto"
    absa_model: str = "yangheng/deberta-v3-base-absa-v1.1"
    absa_allow_model_download: bool = False
    absa_max_quadruples: int = 10

    # ---------- Tri-agent ----------
    sdi_threshold: float = 0.45
    finbert_model: str = "ProsusAI/finbert"
    enable_finbert: bool = True
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_input_price_per_m: float = 0.27
    llm_output_price_per_m: float = 1.10

    # ---------- LLM ----------
    llm_provider: str = "deepseek"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # ---------- alerts ----------
    alert_window_seconds: int = 300
    alert_min_samples: int = 8
    alert_negative_ratio: float = 0.6
    alert_cooldown_seconds: int = 120

    # ---------- API ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @field_validator("data_dir", "reports_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @field_validator("sdi_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("SDI_THRESHOLD must be within [0, 1]")
        return value

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.reports_dir, self.data_dir / "logs"):
            path.mkdir(parents=True, exist_ok=True)

    @property
    def llm_enabled(self) -> bool:
        return bool(self.deepseek_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()
