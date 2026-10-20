from __future__ import annotations

from src.absa_engine import ABSAEngine, find_aspects, find_entities
from src.schemas import SentimentLabel


def test_chinese_absa_quadruples(test_settings):
    engine = ABSAEngine(test_settings)
    result = engine.analyze("币安提现三天了还没到账，客服也一直敷衍，太失望了。")

    assert result.language == "zh"
    quadruples = result.quadruples
    assert quadruples, "expected at least one quadruple"
    entities = {q.entity for q in quadruples}
    assert "币安" in entities
    aspects = {q.aspect_category for q in quadruples}
    assert {"withdrawal", "customer_service"}.issubset(aspects)
    withdrawal = next(q for q in quadruples if q.aspect_category == "withdrawal")
    assert withdrawal.label is SentimentLabel.NEGATIVE
    assert withdrawal.sentiment < 0
    assert withdrawal.opinion_term
    assert len(withdrawal.quadruple()) == 4


def test_entity_hint_fallback(test_settings):
    engine = ABSAEngine(test_settings)
    result = engine.analyze("提现一直失败，半天不到账。", entity_hint="OKX")
    assert result.quadruples
    assert {q.entity for q in result.quadruples} == {"OKX"}


def test_english_absa(test_settings):
    engine = ABSAEngine(test_settings)
    result = engine.analyze(
        "Coinbase withdrawal is delayed and customer service is terrible."
    )
    assert result.language == "en"
    assert result.quadruples
    assert {q.entity for q in result.quadruples} == {"Coinbase"}
    assert all(q.label is SentimentLabel.NEGATIVE for q in result.quadruples)


def test_entity_and_aspect_detection():
    entities = {name for name, _ in find_entities("OKX 和币安的手续费都太高")}
    assert entities == {"OKX", "币安"}
    aspects = {item[0] for item in find_aspects("提现不到账，手续费还很高")}
    assert "withdrawal" in aspects
    assert "fee" in aspects


def test_empty_text_is_safe(test_settings):
    engine = ABSAEngine(test_settings)
    result = engine.analyze("   ")
    assert result.quadruples == []
    assert result.latency_ms == 0.0
