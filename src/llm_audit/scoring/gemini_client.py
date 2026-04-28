"""GeminiClient — score a résumé prompt via Google AI Studio (Gemini).

Pilot does NOT use this client (Phase 5.4 is Zhipu-only). Main batch
(Phase 5.5) wires this in. Reads GOOGLE_AI_STUDIO_API_KEY from env.

Free-tier limits as of 2026-04: Gemini 2.5 Flash = 15 RPM, 1500 RPD.
Pricing: 0 on free tier (used here); paid tier has separate rates.
"""

from __future__ import annotations

import os
import time
from typing import Any, Final, cast

import google.generativeai as genai

from llm_audit.scoring.base import ScoringClient, ScoringResult
from llm_audit.utils.prompts import (
    SCORING_RESPONSE_JSON_SCHEMA,
    SCORING_SYSTEM_PROMPT,
    ScoringPrompt,
    ScoringResponse,
)

_FREE_TIER_PRICING_USD: Final[float] = 0.0


class GeminiClient(ScoringClient):
    provider = "google_ai_studio"

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GOOGLE_AI_STUDIO_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_AI_STUDIO_API_KEY not set; load .env or pass api_key=...")
        self.model_id = model_id
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=SCORING_SYSTEM_PROMPT,
        )

    def score(self, prompt: ScoringPrompt, cell_id: int = 0) -> ScoringResult:
        t0 = time.monotonic()
        resp = self._model.generate_content(
            prompt.user_message,
            generation_config=cast(
                Any,
                {
                    "response_mime_type": "application/json",
                    "response_schema": SCORING_RESPONSE_JSON_SCHEMA,
                    "temperature": 0.0,
                },
            ),
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        text = resp.text or "{}"
        parsed = ScoringResponse.model_validate_json(text)

        usage = getattr(resp, "usage_metadata", None)
        tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
        tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0)

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
