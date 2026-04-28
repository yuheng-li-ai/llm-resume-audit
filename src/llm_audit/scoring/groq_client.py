"""GroqClient — score a résumé prompt via Groq (Llama 3.3 70B).

Pilot does NOT use this client (Phase 5.4 is Zhipu-only). Main batch
(Phase 5.5) wires this in. Reads GROQ_API_KEY from env.

Free-tier limits as of 2026-04: Llama 3.3 70B Versatile = 30 RPM and
roughly 500K tokens/day. Pricing: 0 on free tier.

Structured output: Groq supports OpenAI-compatible response_format
{"type": "json_object"}. The system prompt explicitly asks for JSON,
which Groq's JSON mode then enforces.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final, cast

from groq import Groq

from llm_audit.scoring.base import ScoringClient, ScoringResult
from llm_audit.utils.prompts import (
    SCORING_SYSTEM_PROMPT,
    ScoringPrompt,
    ScoringResponse,
)

_FREE_TIER_PRICING_USD: Final[float] = 0.0


class GroqClient(ScoringClient):
    provider = "groq"

    def __init__(
        self,
        model_id: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set; load .env or pass api_key=...")
        self.model_id = model_id
        self._client = Groq(api_key=key)

    def score(self, prompt: ScoringPrompt, cell_id: int = 0) -> ScoringResult:
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=cast(
                Any,
                [
                    {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt.user_message},
                ],
            ),
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        choice = resp.choices[0]
        text = choice.message.content or "{}"
        parsed = ScoringResponse.model_validate_json(text)

        usage = resp.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)

        return ScoringResult(
            cell_id=cell_id,
            model_id=self.model_id,
            provider=self.provider,
            hiring_score=parsed.hiring_score,
            rationale=parsed.rationale,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_local=_FREE_TIER_PRICING_USD,
            currency="USD",
            raw_response=text,
        )

    def estimate_cost(
        self,
        n_calls: int,
        avg_tokens_in: int,
        avg_tokens_out: int,
    ) -> float:
        return _FREE_TIER_PRICING_USD * n_calls
