"""Tests for ScoringClient ABC + ScoringResult (Phase 5.2)."""

from __future__ import annotations

import pytest

from llm_audit.scoring.base import ScoringClient, ScoringResult
from llm_audit.utils.prompts import ScoringPrompt


class _FakeClient(ScoringClient):
    """Minimal subclass for ABC contract testing — no real API calls."""

    provider = "fake"
    model_id = "fake-model-1"

    def score(self, prompt: ScoringPrompt, cell_id: int = 0) -> ScoringResult:
        return ScoringResult(
            cell_id=cell_id,
            model_id=self.model_id,
            provider=self.provider,
            hiring_score=50,
            rationale="ok",
            latency_ms=10,
            tokens_in=100,
            tokens_out=20,
            cost_local=0.0,
            currency="CNY",
            raw_response='{"hiring_score": 50, "rationale": "ok"}',
        )

    def estimate_cost(self, n_calls: int, avg_tokens_in: int, avg_tokens_out: int) -> float:
        return 0.0


class TestABCContract:
    def test_cannot_instantiate_base(self) -> None:
        with pytest.raises(TypeError):
            ScoringClient()  # type: ignore[abstract]

    def test_subclass_with_score_can_instantiate(self) -> None:
        c = _FakeClient()
        assert c.provider == "fake"
        assert c.model_id == "fake-model-1"

    def test_score_returns_scoring_result(self) -> None:
        c = _FakeClient()
        r = c.score(ScoringPrompt(job_description="x", resume_text="y"))
        assert isinstance(r, ScoringResult)
        assert r.hiring_score == 50
        assert r.model_id == "fake-model-1"


class TestScoringResult:
    def test_constructs_with_required_fields(self) -> None:
        r = ScoringResult(
            cell_id=42,
            model_id="glm-4-flash",
            provider="zhipu",
            hiring_score=80,
            rationale="Strong match.",
            latency_ms=350,
            tokens_in=900,
            tokens_out=80,
            cost_local=0.0,
            currency="CNY",
            raw_response='{"hiring_score": 80}',
        )
        assert r.cell_id == 42

    def test_is_frozen(self) -> None:
        r = ScoringResult(
            cell_id=0,
            model_id="m",
            provider="p",
            hiring_score=50,
            rationale="ok",
            latency_ms=10,
            tokens_in=10,
            tokens_out=5,
            cost_local=0.0,
            currency="CNY",
            raw_response="{}",
        )
        with pytest.raises((AttributeError, TypeError)):
            r.hiring_score = 99  # type: ignore[misc]
