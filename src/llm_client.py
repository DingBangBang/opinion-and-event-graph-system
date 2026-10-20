"""Minimal OpenAI-compatible chat client (DeepSeek by default).

Kept dependency-light on purpose: httpx + tenacity instead of a vendor SDK, so
the same client also works with OpenAI / Moonshot / Qwen compatible endpoints.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLMError(RuntimeError):
    pass


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    def cost_usd(self, input_price_per_m: float, output_price_per_m: float) -> float:
        return (
            self.prompt_tokens / 1_000_000 * input_price_per_m
            + self.completion_tokens / 1_000_000 * output_price_per_m
        )


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage


class LLMClient:
    """Thin sync client with retries and defensive JSON extraction."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        timeout: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, *, temperature: float = 0.1) -> LLMResponse:
        if not self.enabled:
            raise LLMError("LLM API key is not configured")

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type((httpx.HTTPError, LLMError)),
            reraise=True,
        )
        def _call() -> LLMResponse:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
                body = response.json()
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
                raise LLMError(f"Malformed LLM response: {body}") from exc
            usage = body.get("usage") or {}
            return LLMResponse(
                content=content,
                usage=LLMUsage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    model=body.get("model", self.model),
                ),
            )

        try:
            return _call()
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            logger.warning("LLM call failed after retries: %s", exc)
            raise LLMError(str(exc)) from exc

    @staticmethod
    def parse_json(content: str) -> dict[str, Any]:
        """LLMs occasionally wrap JSON in markdown fences; handle both."""

        text = content.strip()
        fenced = _JSON_BLOCK.search(text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                parsed = json.loads(text[start:end + 1])
            else:
                raise
        if not isinstance(parsed, dict):
            raise ValueError("LLM JSON root must be an object")
        return parsed
