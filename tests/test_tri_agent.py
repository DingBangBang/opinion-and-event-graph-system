from __future__ import annotations

from src.schemas import AgentVerdict, SentimentLabel, label_from_score
from src.tri_agent import TriAgentRouter


class FakeAgent:
    def __init__(self, name: str, score: float, confidence: float = 0.9):
        self.name = name
        self._score = score
        self._confidence = confidence

    def analyze(self, text: str) -> AgentVerdict:
        return AgentVerdict(
            agent=self.name,
            score=self._score,
            label=label_from_score(self._score),
            confidence=self._confidence,
            latency_ms=0.1,
            rationale="test double",
        )


def test_sdi_formula():
    vader = AgentVerdict(
        agent="vader", score=-0.8, label=SentimentLabel.NEGATIVE, confidence=0.9
    )
    finbert = AgentVerdict(
        agent="finbert", score=0.2, label=SentimentLabel.POSITIVE, confidence=0.8
    )
    sdi = TriAgentRouter.semantic_divergence(vader, finbert)
    # 0.7 * |(-0.8 - 0.2)| / 2 + 0.3 * 1 = 0.65
    assert sdi == 0.65
    identical = TriAgentRouter.semantic_divergence(vader, vader)
    assert identical == 0.0


def test_near_consensus_skips_llm(test_settings):
    router = TriAgentRouter(
        test_settings,
        vader=FakeAgent("vader", -0.6, 0.9),
        finbert=FakeAgent("finbert", -0.55, 0.8),
    )
    decision = router.analyze("币安提现失败，客服敷衍。", comment_id="t-consensus")
    assert decision.routed_to_llm is False
    assert decision.selected_agent in {"vader", "finbert"}
    assert decision.llm_score is None
    assert decision.avoided_cost_usd > 0
    assert decision.estimated_cost_usd == 0
    records = router.audit.read_records()
    assert records and records[-1].comment_id == "t-consensus"


def test_divergence_routes_to_llm_fallback(test_settings):
    router = TriAgentRouter(
        test_settings,
        vader=FakeAgent("vader", -0.9, 0.9),
        finbert=FakeAgent("finbert", 0.8, 0.9),
    )
    decision = router.analyze(
        "客服很敷衍，但提现秒到，真不知道该夸还是该骂。", comment_id="t-divergent"
    )
    assert decision.sdi > test_settings.sdi_threshold
    assert decision.routed_to_llm is True
    assert decision.selected_agent == "llm"
    assert decision.llm_score is not None
    # No API key in tests: heuristic fallback keeps the pipeline alive at zero cost.
    assert decision.estimated_cost_usd == 0
    assert len(decision.agents) == 3


def test_sdi_threshold_is_respected(test_settings):
    strict = test_settings.model_copy(update={"sdi_threshold": 0.99})
    router = TriAgentRouter(
        strict,
        vader=FakeAgent("vader", -0.9, 0.9),
        finbert=FakeAgent("finbert", 0.8, 0.9),
    )
    decision = router.analyze("测试阈值", comment_id="t-threshold")
    assert decision.routed_to_llm is False
