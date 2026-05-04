"""ZhipuClient — score a résumé prompt via Zhipu's GLM family.

Supports two model_ids:
    - "glm-4-flash"   (free tier, used for the pilot)
    - "glm-5"         (paid tier, used in main batch for the GLM 5.1 row)

Structured JSON output is requested via the `tools` (function calling)
mechanism, which Zhipu's SDK exposes as part of chat.completions.create.

This module makes real LLM API calls inside `score()`. Reads
ZHIPUAI_API_KEY from the environment (loaded by python-dotenv at the
caller's top level).
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Final, cast

from zhipuai import ZhipuAI

from llm_audit.scoring.base import ScoringClient, ScoringResult
from llm_audit.utils.prompts import (
    SCORING_RESPONSE_JSON_SCHEMA,
    SCORING_SYSTEM_PROMPT,
    ScoringPrompt,
    ScoringResponse,
)

# Zhipu pricing per million tokens (CNY). Free models = 0.
# Source: https://open.bigmodel.cn/pricing as of 2026-04.
_PRICING_CNY_PER_MILLION_TOKENS: Final[dict[str, tuple[float, float]]] = {
    # model_id: (input, output)
    "glm-4-flash": (0.0, 0.0),
    "glm-4-flash-250414": (0.0, 0.0),
    "glm-5": (5.0, 20.0),
    "glm-4.5": (2.0, 8.0),
}

_TOOL_NAME: Final[str] = "submit_hiring_score"


def _strip_markdown_fence(text: str) -> str:
    """GLM-4 Flash often wraps JSON in ```json ... ``` fences instead of
    honouring tool-call mode. Strip them before parsing."""
    s = text.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else s


def _unwrap_tool_envelope(text: str) -> str:
    """Some Zhipu replies double-wrap the args under the tool name:
        {"submit_hiring_score": {"hiring_score": 86, "rationale": "..."}}
    Unwrap to the inner object if it matches our schema shape.
    """
    import json

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return text
    if (
        isinstance(obj, dict)
        and len(obj) == 1
        and _TOOL_NAME in obj
        and isinstance(obj[_TOOL_NAME], dict)
    ):
        return json.dumps(obj[_TOOL_NAME])
    return text


class ZhipuClient(ScoringClient):
    provider = "zhipu"

    def __init__(
        self,
        model_id: str = "glm-4-flash",
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("ZHIPUAI_API_KEY")
        if not key:
            raise RuntimeError("ZHIPUAI_API_KEY not set; load .env or pass api_key=...")
        self.model_id = model_id
        self._timeout = timeout
        self._client = ZhipuAI(api_key=key, timeout=timeout)

    def score(self, prompt: ScoringPrompt, cell_id: int = 0) -> ScoringResult:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": "Submit the hiring score and rationale for the candidate.",
                    "parameters": SCORING_RESPONSE_JSON_SCHEMA,
                },
            }
        ]
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SCORING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt.user_message},
            ],
            tools=cast(Any, tools),
            tool_choice=cast(Any, {"type": "function", "function": {"name": _TOOL_NAME}}),
            temperature=0.0,
            timeout=self._timeout,
        )
        latency_ms = int((time.monotonic() - t0) * 1000)

        choice = resp.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            args_str = tool_calls[0].function.arguments
        else:
            args_str = message.content or "{}"
        cleaned = _strip_markdown_fence(args_str)
        unwrapped = _unwrap_tool_envelope(cleaned)
        parsed = ScoringResponse.model_validate_json(unwrapped)

        usage = resp.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = self._compute_cost(tokens_in, tokens_out)

        return ScoringResult(
            cell_id=cell_id,
            model_id=self.model_id,
            provider=self.provider,
            hiring_score=parsed.hiring_score,
            rationale=parsed.rationale,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_local=cost,
            currency="CNY",
            raw_response=args_str,
        )

    def estimate_cost(
        self,
        n_calls: int,
        avg_tokens_in: int,
        avg_tokens_out: int,
    ) -> float:
        per_in, per_out = _PRICING_CNY_PER_MILLION_TOKENS.get(self.model_id, (0.0, 0.0))
        total_in_m = n_calls * avg_tokens_in / 1_000_000
        total_out_m = n_calls * avg_tokens_out / 1_000_000
        return per_in * total_in_m + per_out * total_out_m

    def _compute_cost(self, tokens_in: int, tokens_out: int) -> float:
        per_in, per_out = _PRICING_CNY_PER_MILLION_TOKENS.get(self.model_id, (0.0, 0.0))
        return (tokens_in * per_in + tokens_out * per_out) / 1_000_000
