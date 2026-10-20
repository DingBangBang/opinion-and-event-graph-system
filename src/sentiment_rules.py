"""Deterministic, dependency-light sentiment scoring rules.

Used by three call sites:

* ``ABSAEngine`` offline backend (opinion-term polarity),
* ``VADERAgent`` for Chinese/mixed text (VADER itself is English-only),
* ``FinBERTAgent`` offline surrogate for finance sentences.

The implementation follows the classic VADER recipe: longest-match lexicon
lookup, then negation and intensifier modifiers inside a short look-back window.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .resources.lexicons import (
    FINANCE_NEGATIVE,
    FINANCE_POSITIVE,
    FINANCE_ZH_NEGATIVE,
    FINANCE_ZH_POSITIVE,
    INTENSIFIERS,
    NEGATIONS,
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
)

FINANCE_TERMS: frozenset[str] = frozenset(
    term.lower()
    for term in (
        FINANCE_ZH_POSITIVE + FINANCE_ZH_NEGATIVE + FINANCE_POSITIVE + FINANCE_NEGATIVE
    )
)

_CLAUSE_RE = re.compile(r"[^，,。！!？?；;\n]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+|[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class LexiconHit:
    term: str
    weight: float
    start: int
    end: int
    negated: bool = False
    intensifier: float = 1.0

    @property
    def effective(self) -> float:
        return (-self.weight if self.negated else self.weight) * self.intensifier


@dataclass
class Clause:
    text: str
    start: int
    end: int


@dataclass
class LexiconScore:
    score: float
    confidence: float
    hits: list[LexiconHit] = field(default_factory=list)


def split_clauses(text: str) -> list[Clause]:
    """Split on Chinese/English punctuation while keeping character offsets."""

    clauses: list[Clause] = []
    for match in _CLAUSE_RE.finditer(text):
        chunk = match.group().strip()
        if chunk:
            offset = match.start() + (len(match.group()) - len(match.group().lstrip()))
            clauses.append(Clause(chunk, offset, offset + len(chunk)))
    return clauses or [Clause(text.strip(), 0, len(text.strip()))]


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def _lookback(text: str, start: int, window: int = 5) -> str:
    return text[max(0, start - window):start].lower()


def _has_negation(prefix: str) -> bool:
    return any(neg in prefix for neg in NEGATIONS)


def _intensity(prefix: str) -> float:
    value = 1.0
    for token, factor in INTENSIFIERS.items():
        if token in prefix:
            value = max(value, factor)
    return value


def lexicon_hits(text: str, *, finance: bool = False) -> list[LexiconHit]:
    """Longest-match-first scan; English aliases are matched case-insensitively."""

    lowered = text.lower()
    lexicon: dict[str, float] = {}
    for term, weight in POSITIVE_TERMS.items():
        lexicon[term.lower()] = abs(weight)
    for term, weight in NEGATIVE_TERMS.items():
        lexicon[term.lower()] = -abs(weight)
    if finance:
        for term in FINANCE_POSITIVE:
            lexicon.setdefault(term.lower(), 0.6)
        for term in FINANCE_NEGATIVE:
            lexicon.setdefault(term.lower(), -0.7)

    hits: list[LexiconHit] = []
    occupied: list[tuple[int, int]] = []
    for term in sorted(lexicon, key=len, reverse=True):
        weight = lexicon[term]
        start = 0
        while True:
            idx = lowered.find(term, start)
            if idx < 0:
                break
            end = idx + len(term)
            overlaps = any(idx < e and end > s for s, e in occupied)
            if not overlaps:
                prefix = _lookback(text, idx)
                hits.append(
                    LexiconHit(
                        term=text[idx:end],
                        weight=weight,
                        start=idx,
                        end=end,
                        negated=_has_negation(prefix),
                        intensifier=_intensity(prefix),
                    )
                )
                occupied.append((idx, end))
            start = end
    hits.sort(key=lambda h: h.start)
    return hits


def score_lexicon(text: str, *, finance: bool = False) -> LexiconScore:
    hits = lexicon_hits(text, finance=finance)
    if not hits:
        return LexiconScore(0.0, 0.2, [])
    raw = sum(hit.effective for hit in hits)
    score = math.tanh(raw / 2.0)
    coverage = min(1.0, len(hits) / 4.0)
    confidence = min(0.95, 0.35 + 0.45 * coverage + 0.15 * min(1.0, abs(score)))
    return LexiconScore(score=round(score, 4), confidence=round(confidence, 4), hits=hits)


def score_finance_sentence(text: str) -> LexiconScore:
    """FinBERT surrogate: only financially salient words count.

    This intentionally ignores generic emotional vocabulary, producing the same
    kind of disagreement a domain model exhibits when it lacks world knowledge
    about service complaints.
    """

    hits = [hit for hit in lexicon_hits(text, finance=True) if hit.term.lower() in FINANCE_TERMS]
    if not hits:
        return LexiconScore(0.0, 0.2, [])
    raw = sum(hit.effective for hit in hits)
    score = math.tanh(raw / 1.8)
    confidence = min(0.9, 0.45 + 0.12 * len(hits))
    return LexiconScore(score=round(score, 4), confidence=round(confidence, 4), hits=hits)


def nearest_hit(hits: list[LexiconHit], start: int, end: int) -> LexiconHit | None:
    """Pick the sentiment word closest to an aspect span."""

    if not hits:
        return None
    return min(hits, key=lambda h: min(abs(h.start - end), abs(h.end - start)))


def opinion_phrase(hit: LexiconHit | None, clause: str, aspect_start: int, aspect_end: int) -> str:
    """Return the lexically strongest opinion phrase, with a safe fallback."""

    if hit is not None:
        return hit.term
    prefix = clause[max(0, aspect_start - 8):aspect_start].strip("的了是在我你他她它")
    if 2 <= len(prefix) <= 8:
        return prefix
    suffix = clause[aspect_end:aspect_end + 6].strip("的了是在我你他她它")
    if 2 <= len(suffix) <= 6:
        return suffix
    return "整体评价"
