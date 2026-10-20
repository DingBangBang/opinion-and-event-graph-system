"""Multi-Agent committee with Semantic Divergence Index (SDI) routing — dimension 2.

Flow
----
VADER (word level) + FinBERT (sentence level)
        │
        ▼
SDI = 0.7 * |vader - finbert| / 2 + 0.3 * label_disagreement
        │
        ├── SDI <= threshold → adopt the higher-confidence small model (cost ≈ 0)
        └── SDI >  threshold → LLM arbitration (cross-clause reasoning)

Every decision is persisted (JSONL + optional Elasticsearch) for offline audit
and for the API's /routing/stats endpoint.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import Settings, settings as default_settings

from .absa_engine import ABSAResult
from .llm_client import LLMClient, LLMError
from .schemas import (
    AgentVerdict,
    RoutingAuditRecord,
    SentimentLabel,
    TriAgentDecision,
    label_from_score,
)
from .sentiment_rules import score_finance_sentence, score_lexicon

logger = logging.getLogger(__name__)


def _is_mostly_english(text: str) -> bool:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    return latin > 0 and cjk * 3 < latin


class VADERAgent:
    """Fast word-level scoring.

    ``vaderSentiment`` only supports English, so Chinese comments are handled by
    the bilingual VADER-style rule engine in :mod:`src.sentiment_rules`
    (lexicon + negation + intensifier). The agent identity and latency budget
    stay the same, which keeps routing comparable across languages.
    """

    name = "vader"

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> AgentVerdict:
        started = time.perf_counter()
        if _is_mostly_english(text):
            scores = self._analyzer.polarity_scores(text)
            compound = float(scores["compound"])
            confidence = min(0.95, 0.45 + 0.5 * abs(compound))
            metadata: dict[str, Any] = {"engine": "vaderSentiment", **scores}
        else:
            result = score_lexicon(text)
            compound = result.score
            confidence = result.confidence
            metadata = {
                "engine": "zh-vader-lexicon",
                "hits": [hit.term for hit in result.hits],
            }
        return AgentVerdict(
            agent="vader",
            score=round(compound, 4),
            label=label_from_score(compound),
            confidence=round(confidence, 4),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rationale="word-level polarity with negation/intensifier modifiers",
            metadata=metadata,
        )


class FinBERTAgent:
    """Domain sentence-level scorer (ProsusAI/finbert when available)."""

    name = "finbert"

    def __init__(
        self,
        model_id: str = "ProsusAI/finbert",
        enabled: bool = True,
        allow_download: bool = False,
    ) -> None:
        self.model_id = model_id
        self.enabled = enabled
        self.allow_download = allow_download
        self._classifier = None
        self._id2label: dict[str, str] = {}

    def _load(self) -> None:
        if self._classifier is not None or not self.enabled:
            return
        from pathlib import Path

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        cached = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / ("models--" + self.model_id.replace("/", "--"))
        ).exists()
        if not cached and not self.allow_download:
            raise RuntimeError(f"{self.model_id} is not cached")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, local_files_only=not self.allow_download
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, local_files_only=not self.allow_download
        )
        self._classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            top_k=None,
        )
        self._id2label = {
            str(k): str(v).lower() for k, v in (model.config.id2label or {}).items()
        }

    def _label(self, raw: str) -> SentimentLabel:
        name = self._id2label.get(raw, raw).lower()
        if "pos" in name:
            return SentimentLabel.POSITIVE
        if "neg" in name:
            return SentimentLabel.NEGATIVE
        return SentimentLabel.NEUTRAL

    def analyze(self, text: str) -> AgentVerdict:
        started = time.perf_counter()
        try:
            self._load()
            if self._classifier is not None and _is_mostly_english(text):
                output = self._classifier(text, truncation=True)[0]
                scores = {
                    self._label(item["label"]): float(item["score"]) for item in output
                }
                pos = scores.get(SentimentLabel.POSITIVE, 0.0)
                neg = scores.get(SentimentLabel.NEGATIVE, 0.0)
                score = max(-1.0, min(1.0, pos - neg))
                confidence = max(scores.values()) if scores else 0.5
                return AgentVerdict(
                    agent="finbert",
                    score=round(score, 4),
                    label=label_from_score(score),
                    confidence=round(confidence, 4),
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    rationale="FinBERT sentence-level probabilities",
                    metadata={"engine": "ProsusAI/finbert", "probs": scores},
                )
        except Exception as exc:  # noqa: BLE001 - surrogate is intentional
            logger.debug("FinBERT unavailable, using finance lexicon: %s", exc)

        result = score_finance_sentence(text)
        return AgentVerdict(
            agent="finbert",
            score=round(result.score, 4),
            label=label_from_score(result.score),
            confidence=round(result.confidence, 4),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rationale="finance-domain sentence lexicon (FinBERT surrogate)",
            metadata={"engine": "finance-lexicon", "hits": [h.term for h in result.hits]},
        )


def estimate_llm_tokens(text: str) -> tuple[int, int]:
    """Rough token estimate used only for the counterfactual cost calculation."""

    prompt = 220 + math.ceil(len(text) / 1.5)
    completion = 90
    return prompt, completion


_LLM_SYSTEM = """你是舆情分析委员会的仲裁 Agent，负责处理小模型分歧较大的复杂评论。
结合上下文、反讽、跨句指代与领域知识给出整体情感判断。
只输出 JSON：{"sentiment": -1.0, "confidence": 0.85,
"label": "negative|neutral|positive", "rationale": "不超过80字的依据"}"""


class LLMAgent:
    """Cross-clause reasoning via an OpenAI-compatible API (DeepSeek default)."""

    name = "llm"

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    @property
    def available(self) -> bool:
        return self.client.enabled

    def analyze(
        self,
        text: str,
        *,
        vader: AgentVerdict,
        finbert: AgentVerdict,
        absa: ABSAResult | None = None,
        sdi: float = 0.0,
    ) -> AgentVerdict:
        started = time.perf_counter()
        if not self.available:
            raise LLMError("LLM is not configured")

        quadruples = absa.structured if absa else []
        user = (
            f"评论：{text}\n"
            f"VADER 得分：{vader.score}（置信度 {vader.confidence}）\n"
            f"FinBERT 得分：{finbert.score}（置信度 {finbert.confidence}）\n"
            f"SDI：{sdi:.3f}\n"
            f"ABSA 四元组：{json.dumps(quadruples, ensure_ascii=False)}\n"
            "请给出整体情感（-1 至 1）及依据。"
        )
        response = self.client.chat(_LLM_SYSTEM, user)
        payload = self.client.parse_json(response.content)
        score = max(-1.0, min(1.0, float(payload.get("sentiment", 0.0))))
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.7))))
        return AgentVerdict(
            agent="llm",
            score=round(score, 4),
            label=label_from_score(score),
            confidence=round(confidence, 4),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rationale=str(payload.get("rationale", ""))[:200],
            metadata={
                "model": response.usage.model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
        )


class RoutingAuditLogger:
    """Durable audit trail: JSONL always, Elasticsearch when a store is wired in."""

    def __init__(self, path: Path, store: object | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self._lock = threading.Lock()

    def log(self, comment_id: str, decision: TriAgentDecision) -> RoutingAuditRecord:
        record = RoutingAuditRecord(
            comment_id=comment_id,
            created_at=datetime.now(timezone.utc),
            sdi=decision.sdi,
            threshold=decision.threshold,
            routed_to_llm=decision.routed_to_llm,
            selected_agent=decision.selected_agent,
            vader_score=decision.vader_score,
            finbert_score=decision.finbert_score,
            llm_score=decision.llm_score,
            final_score=decision.final_score,
            estimated_cost_usd=decision.estimated_cost_usd,
            avoided_cost_usd=decision.avoided_cost_usd,
            rationale=decision.rationale,
        )
        line = record.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self.store is not None:
            try:
                self.store.index_decision(record)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - audit must never break the stream
                logger.warning("Failed to index routing audit record: %s", exc)
        return record

    def read_records(self) -> list[RoutingAuditRecord]:
        if not self.path.exists():
            return []
        records: list[RoutingAuditRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        records.append(RoutingAuditRecord.model_validate_json(line))
                    except ValueError:
                        continue
        return records


class TriAgentRouter:
    """Committee coordinator implementing the SDI cost/latency trade-off."""

    def __init__(
        self,
        config: Settings | None = None,
        *,
        llm_client: LLMClient | None = None,
        audit_logger: RoutingAuditLogger | None = None,
        vader: VADERAgent | None = None,
        finbert: FinBERTAgent | None = None,
    ) -> None:
        self.settings = config or default_settings
        self.llm_client = llm_client or LLMClient(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
        )
        self.vader = vader or VADERAgent()
        self.finbert = finbert or FinBERTAgent(
            model_id=self.settings.finbert_model,
            enabled=self.settings.enable_finbert,
            allow_download=self.settings.absa_allow_model_download,
        )
        self.llm = LLMAgent(self.llm_client)
        self.audit = audit_logger or RoutingAuditLogger(
            self.settings.data_dir / "routing_audit.jsonl"
        )

    @staticmethod
    def semantic_divergence(vader: AgentVerdict, finbert: AgentVerdict) -> float:
        """SDI = 0.7 * normalized score gap + 0.3 * label disagreement."""

        gap = abs(vader.score - finbert.score) / 2.0
        disagreement = 0.0 if vader.label is finbert.label else 1.0
        return round(min(1.0, 0.7 * gap + 0.3 * disagreement), 4)

    def _llm_cost(self, verdict: AgentVerdict) -> float:
        prompt = float(verdict.metadata.get("prompt_tokens", 0))
        completion = float(verdict.metadata.get("completion_tokens", 0))
        return (
            prompt / 1_000_000 * self.settings.llm_input_price_per_m
            + completion / 1_000_000 * self.settings.llm_output_price_per_m
        )

    def _counterfactual_cost(self, text: str) -> float:
        prompt, completion = estimate_llm_tokens(text)
        return (
            prompt / 1_000_000 * self.settings.llm_input_price_per_m
            + completion / 1_000_000 * self.settings.llm_output_price_per_m
        )

    def _heuristic_arbitration(
        self,
        text: str,
        vader: AgentVerdict,
        finbert: AgentVerdict,
        absa: ABSAResult | None,
    ) -> AgentVerdict:
        """Offline fallback so the pipeline still works without an API key."""

        started = time.perf_counter()
        absa_scores = [q.sentiment for q in absa.quadruples] if absa else []
        absa_mean = sum(absa_scores) / len(absa_scores) if absa_scores else 0.0
        weights = [0.35, 0.4] + ([0.25] if absa_scores else [])
        values = [vader.score, finbert.score] + ([absa_mean] if absa_scores else [])
        score = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        return AgentVerdict(
            agent="llm",
            score=round(max(-1.0, min(1.0, score)), 4),
            label=label_from_score(score),
            confidence=0.55 if absa_scores else 0.45,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            rationale="LLM 不可用，使用 VADER/FinBERT/ABSA 加权共识兜底",
            metadata={"engine": "heuristic-fallback", "absa_mean": round(absa_mean, 4)},
        )

    def analyze(
        self,
        text: str,
        *,
        absa: ABSAResult | None = None,
        comment_id: str | None = None,
    ) -> TriAgentDecision:
        started = time.perf_counter()
        text = (text or "").strip()
        comment_id = comment_id or f"adhoc-{uuid.uuid4().hex[:10]}"

        vader = self.vader.analyze(text)
        finbert = self.finbert.analyze(text)
        sdi = self.semantic_divergence(vader, finbert)
        routed = sdi > self.settings.sdi_threshold

        llm_verdict: AgentVerdict | None = None
        estimated_cost = 0.0
        avoided_cost = 0.0

        if routed:
            try:
                llm_verdict = self.llm.analyze(
                    text, vader=vader, finbert=finbert, absa=absa, sdi=sdi
                )
                estimated_cost = self._llm_cost(llm_verdict)
            except Exception as exc:  # noqa: BLE001 - offline fallback
                logger.debug("LLM arbitration unavailable (%s); using fallback", exc)
                llm_verdict = self._heuristic_arbitration(text, vader, finbert, absa)

            selected: str = "llm"
            final_score = llm_verdict.score
            confidence = llm_verdict.confidence
            rationale = (
                f"SDI={sdi:.3f} > {self.settings.sdi_threshold:.2f}，升级 LLM 仲裁；"
                f"{llm_verdict.rationale}"
            )
        else:
            chosen = vader if vader.confidence >= finbert.confidence else finbert
            selected = chosen.agent
            final_score = chosen.score
            confidence = chosen.confidence
            avoided_cost = self._counterfactual_cost(text)
            rationale = (
                f"SDI={sdi:.3f} <= {self.settings.sdi_threshold:.2f}，"
                f"采纳高置信小模型 {chosen.agent}，跳过 LLM"
            )

        agents = [vader, finbert] + ([llm_verdict] if llm_verdict else [])
        decision = TriAgentDecision(
            vader_score=vader.score,
            finbert_score=finbert.score,
            llm_score=llm_verdict.score if llm_verdict else None,
            sdi=sdi,
            threshold=self.settings.sdi_threshold,
            routed_to_llm=routed,
            selected_agent=selected,  # type: ignore[arg-type]
            final_score=round(final_score, 4),
            final_label=label_from_score(final_score),
            confidence=round(confidence, 4),
            rationale=rationale,
            estimated_cost_usd=round(estimated_cost, 8),
            avoided_cost_usd=round(avoided_cost, 8),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            agents=agents,
        )
        self.audit.log(comment_id, decision)
        return decision

    def analyze_batch(
        self, items: list[tuple[str, ABSAResult | None, str]]
    ) -> list[TriAgentDecision]:
        return [
            self.analyze(text, absa=absa, comment_id=comment_id)
            for text, absa, comment_id in items
        ]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .absa_engine import ABSAEngine

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    engine = ABSAEngine()
    router = TriAgentRouter()
    samples = [
        "币安提现三天了还没到账，客服也一直敷衍，太失望了。",
        "OKX 的 App 很流畅，提现秒到，手续费也低，体验不错。",
        "火币HTX 被盗号之后客服不肯处理，资金冻结，准备投诉监管。",
    ]
    for sample in samples:
        absa_result = engine.analyze(sample)
        decision = router.analyze(sample, absa=absa_result)
        print(f"\n{sample}\n  ABSA: {[q.quadruple() for q in absa_result.quadruples]}")
        print(
            f"  VADER={decision.vader_score:+.2f} FinBERT={decision.finbert_score:+.2f} "
            f"SDI={decision.sdi:.2f} routed={decision.routed_to_llm} "
            f"→ {decision.selected_agent} ({decision.final_score:+.2f})"
        )

