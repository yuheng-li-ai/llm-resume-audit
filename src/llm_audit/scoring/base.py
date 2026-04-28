"""ScoringClient ABC + ScoringResult dataclass (Phase 5.2).

Provider-specific subclasses (currently ZhipuClient for GLM 5.1 and GLM 4.5)
implement `score(prompt, cell_id) -> ScoringResult` and
`estimate_cost(n_calls, avg_tokens_in, avg_tokens_out) -> float` using
their own SDK and pricing.

Subclasses MUST set class-level attributes `provider` (vendor identifier)
and `model_id` (specific model name), and MUST NOT make any LLM API
call inside __init__ — instantiation should be free of network IO.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_audit.utils.prompts import ScoringPrompt


@dataclass(frozen=True)
class ScoringResult:
    """One scored cell × model observation."""

    cell_id: int
    model_id: str
    provider: str
    hiring_score: int
    rationale: str
    latency_ms: int
    tokens_in: int
    tokens_out: int
    cost_local: float
    currency: str
    raw_response: str


class ScoringClient(ABC):
    """Abstract base class for provider-specific scoring clients."""

    provider: str
    model_id: str

    @abstractmethod
    def score(self, prompt: ScoringPrompt, cell_id: int = 0) -> ScoringResult: ...

    @abstractmethod
    def estimate_cost(
        self,
        n_calls: int,
        avg_tokens_in: int,
        avg_tokens_out: int,
    ) -> float: ...
