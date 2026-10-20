"""Aspect-Based Sentiment Analysis (ABSA) engine — dimension 1.

Backends
--------
``hf``       HuggingFace ``yangheng/deberta-v3-base-absa-v1.1`` (sentence + aspect pair)
``deepseek`` DeepSeek / OpenAI-compatible JSON-mode extraction
``lexicon``  deterministic bilingual fallback (default in CI / offline demo)

Every backend emits the same structured quadruple:
``(entity, aspect_category, opinion_term, sentiment_label)`` plus a continuous
score, confidence and evidence span.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from config.settings import Settings, settings as default_settings

from .llm_client import LLMClient, LLMError
from .resources.lexicons import ASPECT_KEYWORDS, ENTITY_ALIASES
from .schemas import AspectSentiment, SentimentLabel, label_from_score
from .sentiment_rules import (
    lexicon_hits,
    nearest_hit,
    opinion_phrase,
    score_lexicon,
    split_clauses,
)

logger = logging.getLogger(__name__)


@dataclass
class ABSAResult:
    text: str
    quadruples: list[AspectSentiment] = field(default_factory=list)
    backend: str = "lexicon"
    language: str = "zh"
    latency_ms: float = 0.0

    @property
    def structured(self) -> list[dict[str, object]]:
        return [
            {
                "entity": q.entity,
                "aspect_category": q.aspect_category,
                "opinion_term": q.opinion_term,
                "sentiment": q.label.value,
                "score": q.sentiment,
            }
            for q in self.quadruples
        ]


# Alias list sorted longest-first so "binance app" wins over "binance".
_ALIAS_INDEX: list[tuple[str, str, str]] = sorted(
    (
        (alias.lower(), canonical, entity_type)
        for canonical, (aliases, entity_type) in ENTITY_ALIASES.items()
        for alias in aliases
    ),
    key=lambda item: len(item[0]),
    reverse=True,
)


def detect_language(text: str) -> str:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_letters = len(re.findall(r"[A-Za-z]", text))
    if cjk and cjk >= ascii_letters * 0.3:
        return "zh"
    if ascii_letters:
        return "en"
    return "zh"


def canonical_entity(name: str) -> tuple[str, str]:
    lowered = name.strip().lower()
    for alias, canonical, entity_type in _ALIAS_INDEX:
        if alias == lowered:
            return canonical, entity_type
    return name.strip(), "unknown"


def find_entities(text: str, hint: str | None = None) -> list[tuple[str, str]]:
    """Return de-duplicated ``(canonical, entity_type)`` in first-mention order."""

    lowered = text.lower()
    matches: list[tuple[int, str, str]] = []
    for alias, canonical, entity_type in _ALIAS_INDEX:
        if alias.isascii():
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            for match in re.finditer(pattern, lowered):
                matches.append((match.start(), canonical, entity_type))
        else:
            start = lowered.find(alias)
            while start >= 0:
                matches.append((start, canonical, entity_type))
                start = lowered.find(alias, start + len(alias))

    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for _, canonical, entity_type in sorted(matches):
        if canonical not in seen:
            seen.add(canonical)
            result.append((canonical, entity_type))

    if not result and hint:
        canonical, entity_type = canonical_entity(hint)
        if canonical:
            result.append((canonical, entity_type or "unknown"))
    return result


def find_aspects(clause: str) -> list[tuple[str, str, int, int]]:
    """Return ``(category, label, start, end)`` for aspects mentioned in a clause."""

    lowered = clause.lower()
    found: dict[str, tuple[str, str, int, int]] = {}
    for category, (label, keywords) in ASPECT_KEYWORDS.items():
        best: tuple[str, int, int] | None = None
        for keyword in keywords:
            idx = lowered.find(keyword.lower())
            if idx >= 0 and (best is None or len(keyword) > len(best[0])):
                best = (keyword, idx, idx + len(keyword))
        if best:
            found[category] = (category, label, best[1], best[2])

    # "整体体验" is only a fallback when no concrete aspect was mentioned.
    if len(found) > 1:
        found.pop("general", None)
    return sorted(found.values(), key=lambda item: item[2])


def _rule_based(
    text: str,
    hint: str | None,
    max_quadruples: int,
) -> list[AspectSentiment]:
    doc_entities = find_entities(text, hint)
    if not doc_entities:
        doc_entities = [("未知实体", "unknown")]

    results: dict[tuple[str, str, str], AspectSentiment] = {}
    for clause in split_clauses(text):
        entities = find_entities(clause.text) or doc_entities
        aspects = find_aspects(clause.text)
        hits = lexicon_hits(clause.text)
        clause_score = score_lexicon(clause.text)

        if not aspects:
            if not hits:
                continue
            aspects = [("general", "整体体验", 0, min(len(clause.text), 6))]

        for entity, entity_type in entities:
            for category, label, aspect_start, aspect_end in aspects:
                hit = nearest_hit(hits, aspect_start, aspect_end)
                if hit is not None:
                    score = max(-1.0, min(1.0, hit.effective))
                    confidence = min(0.92, 0.55 + 0.3 * abs(score))
                    opinion = opinion_phrase(hit, clause.text, aspect_start, aspect_end)
                else:
                    score = clause_score.score
                    confidence = max(0.25, min(0.6, clause_score.confidence - 0.15))
                    opinion = opinion_phrase(None, clause.text, aspect_start, aspect_end)

                key = (entity, category, opinion)
                candidate = AspectSentiment(
                    entity=entity,
                    entity_type=entity_type,
                    aspect_category=category,
                    aspect_label=label,
                    opinion_term=opinion[:20],
                    sentiment=round(score, 4),
                    label=label_from_score(score),
                    confidence=round(confidence, 4),
                    evidence=clause.text[:160],
                )
                previous = results.get(key)
                if previous is None or abs(candidate.sentiment) > abs(previous.sentiment):
                    results[key] = candidate

    ordered = sorted(results.values(), key=lambda q: abs(q.sentiment), reverse=True)
    return ordered[:max_quadruples]


class HFABSABackend:
    """DeBERTa ABSA over rule-based entity/aspect candidates."""

    def __init__(self, model_id: str, allow_download: bool = False, max_quadruples: int = 10):
        self.model_id = model_id
        self.allow_download = allow_download
        self.max_quadruples = max_quadruples
        self._classifier = None
        self._id2label: dict[str, str] = {}

    @staticmethod
    def model_cached(model_id: str) -> bool:
        from pathlib import Path

        folder = "models--" + model_id.replace("/", "--")
        return (Path.home() / ".cache" / "huggingface" / "hub" / folder).exists()

    def _load(self) -> None:
        if self._classifier is not None:
            return
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

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
            top_k=1,
        )
        raw = getattr(model.config, "id2label", {}) or {}
        for key, value in raw.items():
            self._id2label[str(key)] = str(value).lower()

    def _label(self, raw_label: str) -> SentimentLabel:
        name = self._id2label.get(raw_label, raw_label).lower()
        if "pos" in name:
            return SentimentLabel.POSITIVE
        if "neg" in name:
            return SentimentLabel.NEGATIVE
        if "neu" in name:
            return SentimentLabel.NEUTRAL
        # DeBERTa ABSA v1.1 default: LABEL_0 negative, LABEL_1 neutral, LABEL_2 positive
        return {
            "label_0": SentimentLabel.NEGATIVE,
            "label_1": SentimentLabel.NEUTRAL,
            "label_2": SentimentLabel.POSITIVE,
        }.get(name, SentimentLabel.NEUTRAL)

    def analyze(self, text: str, hint: str | None) -> list[AspectSentiment]:
        self._load()
        candidates = _rule_based(text, hint, self.max_quadruples)
        if not candidates:
            return []
        enriched: list[AspectSentiment] = []
        for candidate in candidates:
            try:
                output = self._classifier(
                    text, text_pair=candidate.aspect_label, truncation=True
                )
                item = output[0] if isinstance(output, list) else output
                label = self._label(str(item.get("label", "")))
                score = float(item.get("score", 0.0))
                signed = score if label is SentimentLabel.POSITIVE else (
                    -score if label is SentimentLabel.NEGATIVE else 0.0
                )
                enriched.append(
                    candidate.model_copy(
                        update={
                            "label": label,
                            "sentiment": round(signed, 4),
                            "confidence": round(min(0.99, score), 4),
                        }
                    )
                )
            except Exception as exc:  # noqa: BLE001 - keep the batch alive
                logger.warning("HF ABSA inference failed, keeping rule candidate: %s", exc)
                enriched.append(candidate)
        return enriched


_SYSTEM_PROMPT = """你是金融/加密货币领域的细粒度情感分析专家。
从用户评论中抽取四元组 (实体, 方面类别, 观点词, 情感极性)。
只输出 JSON 对象，格式：
{"quadruples":[{"entity":"币安","aspect_category":"withdrawal",
"opinion_term":"提现失败","sentiment":-0.9,"label":"negative"}]}
要求：
1. aspect_category 必须取自给定类别；无法归类时用 general。
2. sentiment 取值 -1..1（-1 极负面，0 中性，1 极正面）。
3. 每个四元组必须有明确观点词；不要编造评论中不存在的内容。
4. 最多输出 {max_items} 条。"""


class DeepSeekABSABackend:
    """JSON-mode extraction; the LLM handles implicit aspects and cross-clause cases."""

    def __init__(self, client: LLMClient, max_quadruples: int = 10):
        self.client = client
        self.max_quadruples = max_quadruples

    def analyze(self, text: str, hint: str | None) -> list[AspectSentiment]:
        taxonomy = "，".join(
            f"{category}({label})" for category, (label, _) in ASPECT_KEYWORDS.items()
        )
        user = (
            f"评论：{text}\n"
            f"实体提示：{hint or '无'}\n"
            f"可选方面类别：{taxonomy}\n"
            f"最多输出 {self.max_quadruples} 条。"
        )
        response = self.client.chat(
            _SYSTEM_PROMPT.replace("{max_items}", str(self.max_quadruples)), user
        )
        payload = self.client.parse_json(response.content)
        items = payload.get("quadruples", [])
        if not isinstance(items, list):
            raise LLMError("quadruples must be a list")

        results: list[AspectSentiment] = []
        for item in items[: self.max_quadruples]:
            try:
                entity = str(item.get("entity") or hint or "未知实体").strip()
                canonical, entity_type = canonical_entity(entity)
                category = str(item.get("aspect_category") or "general").strip()
                label, aspect_label = ASPECT_KEYWORDS.get(
                    category, (category, category)
                )
                score = float(item.get("sentiment", 0.0))
                score = max(-1.0, min(1.0, score))
                results.append(
                    AspectSentiment(
                        entity=canonical,
                        entity_type=entity_type,
                        aspect_category=label,
                        aspect_label=aspect_label,
                        opinion_term=str(item.get("opinion_term") or "整体评价")[:20],
                        sentiment=round(score, 4),
                        label=label_from_score(score),
                        confidence=0.85,
                        evidence=text[:160],
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.debug("Skipping malformed LLM quadruple %s: %s", item, exc)
        return results


class ABSAEngine:
    """Facade that resolves a backend once and falls back safely."""

    def __init__(
        self,
        config: Settings | None = None,
        backend: str | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = config or default_settings
        self.requested_backend = (backend or self.settings.absa_backend).lower()
        self._llm = llm_client or LLMClient(
            api_key=self.settings.deepseek_api_key,
            base_url=self.settings.deepseek_base_url,
            model=self.settings.deepseek_model,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
        )
        self._hf: HFABSABackend | None = None
        self._deepseek: DeepSeekABSABackend | None = None
        self.backend = self._resolve_backend()

    def _resolve_backend(self) -> str:
        requested = self.requested_backend
        if requested == "lexicon":
            return "lexicon"
        if requested in {"auto", "deepseek"} and self._llm.enabled:
            self._deepseek = DeepSeekABSABackend(
                self._llm, self.settings.absa_max_quadruples
            )
            return "deepseek"
        if requested in {"auto", "hf"}:
            available = False
            try:
                import transformers  # noqa: F401

                available = self.settings.absa_allow_model_download or (
                    HFABSABackend.model_cached(self.settings.absa_model)
                )
            except ImportError:
                available = False
            if available:
                self._hf = HFABSABackend(
                    self.settings.absa_model,
                    allow_download=self.settings.absa_allow_model_download,
                    max_quadruples=self.settings.absa_max_quadruples,
                )
                return "hf"
        if requested in {"hf", "deepseek"}:
            logger.warning(
                "ABSA backend '%s' unavailable (missing key/model); using lexicon",
                requested,
            )
        return "lexicon"

    def analyze(self, text: str, entity_hint: str | None = None) -> ABSAResult:
        started = time.perf_counter()
        text = (text or "").strip()
        backend_used = self.backend
        quadruples: list[AspectSentiment] = []
        if not text:
            return ABSAResult(text="", backend=backend_used, latency_ms=0.0)

        if backend_used == "deepseek" and self._deepseek is not None:
            try:
                quadruples = self._deepseek.analyze(text, entity_hint)
            except Exception as exc:  # noqa: BLE001 - degrade instead of dropping data
                logger.warning("DeepSeek ABSA failed (%s); falling back to lexicon", exc)
                backend_used = "lexicon"
                quadruples = _rule_based(
                    text, entity_hint, self.settings.absa_max_quadruples
                )
        elif backend_used == "hf" and self._hf is not None:
            try:
                quadruples = self._hf.analyze(text, entity_hint)
            except Exception as exc:  # noqa: BLE001
                logger.warning("HF ABSA failed (%s); falling back to lexicon", exc)
                backend_used = "lexicon"
                quadruples = _rule_based(
                    text, entity_hint, self.settings.absa_max_quadruples
                )
        else:
            quadruples = _rule_based(
                text, entity_hint, self.settings.absa_max_quadruples
            )

        if not quadruples:
            quadruples = _rule_based(
                text, entity_hint, self.settings.absa_max_quadruples
            )
            if backend_used != "lexicon":
                backend_used = "lexicon"

        return ABSAResult(
            text=text,
            quadruples=quadruples,
            backend=backend_used,
            language=detect_language(text),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def analyze_batch(
        self, texts: list[str], entity_hints: list[str | None] | None = None
    ) -> list[ABSAResult]:
        hints = entity_hints or [None] * len(texts)
        return [self.analyze(text, hint) for text, hint in zip(texts, hints)]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json
    import sys

    engine = ABSAEngine()
    text = " ".join(sys.argv[1:]) or "币安提现三天还没到账，客服也一直敷衍，太失望了。"
    result = engine.analyze(text)
    print(
        json.dumps(
            {
                "backend": result.backend,
                "language": result.language,
                "latency_ms": result.latency_ms,
                "quadruples": result.structured,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

