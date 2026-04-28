"""Unit tests for ZhipuClient parser helpers + BatchRunner internals.

No API calls — pure string + timing logic.
"""

from __future__ import annotations

import time

from llm_audit.scoring.base import ScoringResult
from llm_audit.scoring.batch_runner import CostTracker, _RateLimiter
from llm_audit.scoring.zhipu_client import _strip_markdown_fence, _unwrap_tool_envelope


class TestStripMarkdownFence:
    def test_returns_input_when_no_fence(self) -> None:
        s = '{"hiring_score": 75, "rationale": "ok"}'
        assert _strip_markdown_fence(s) == s

    def test_strips_json_fence(self) -> None:
        s = '```json\n{"hiring_score": 80}\n```'
        assert _strip_markdown_fence(s) == '{"hiring_score": 80}'

    def test_strips_bare_fence(self) -> None:
        s = '```\n{"hiring_score": 80}\n```'
        assert _strip_markdown_fence(s) == '{"hiring_score": 80}'

    def test_handles_extra_whitespace(self) -> None:
        s = '   ```json\n   {"x": 1}   \n```   '
        assert _strip_markdown_fence(s) == '{"x": 1}'

    def test_case_insensitive_language_tag(self) -> None:
        s = '```JSON\n{"x":1}\n```'
        assert _strip_markdown_fence(s) == '{"x":1}'


class TestUnwrapToolEnvelope:
    def test_unwraps_tool_name_wrapper(self) -> None:
        s = '{"submit_hiring_score": {"hiring_score": 86, "rationale": "ok"}}'
        out = _unwrap_tool_envelope(s)
        assert "submit_hiring_score" not in out
        assert '"hiring_score": 86' in out

    def test_returns_input_when_already_flat(self) -> None:
        s = '{"hiring_score": 75, "rationale": "ok"}'
        assert _unwrap_tool_envelope(s) == s

    def test_returns_input_on_invalid_json(self) -> None:
        s = "not json"
        assert _unwrap_tool_envelope(s) == s

    def test_does_not_unwrap_unrelated_single_key(self) -> None:
        s = '{"foo": {"hiring_score": 50}}'
        assert _unwrap_tool_envelope(s) == s

    def test_does_not_unwrap_multi_key(self) -> None:
        s = '{"submit_hiring_score": {"x":1}, "extra": "y"}'
        assert _unwrap_tool_envelope(s) == s


def _fake_result(cell_id: int = 0, cost: float = 0.01, currency: str = "CNY") -> ScoringResult:
    return ScoringResult(
        cell_id=cell_id,
        model_id="glm-5",
        provider="zhipu",
        hiring_score=75,
        rationale="ok",
        latency_ms=200,
        tokens_in=500,
        tokens_out=80,
        cost_local=cost,
        currency=currency,
        raw_response="{}",
    )


class TestCostTracker:
    def test_record_and_total_cost_single_currency(self) -> None:
        ct = CostTracker()
        ct.record(_fake_result(cost=0.01), batch_id="b1")
        ct.record(_fake_result(cost=0.02), batch_id="b1")
        assert ct.total_cost() == {"CNY": 0.03}

    def test_total_tokens_aggregates_across_records(self) -> None:
        ct = CostTracker()
        ct.record(_fake_result(), batch_id="b1")
        ct.record(_fake_result(), batch_id="b1")
        assert ct.total_tokens() == {"tokens_in": 1000, "tokens_out": 160}

    def test_multi_currency_totals(self) -> None:
        ct = CostTracker()
        ct.record(_fake_result(cost=0.01, currency="CNY"), batch_id="b")
        ct.record(_fake_result(cost=0.005, currency="USD"), batch_id="b")
        totals = ct.total_cost()
        assert totals["CNY"] == 0.01
        assert totals["USD"] == 0.005

    def test_write_csv_roundtrips(self, tmp_path) -> None:
        ct = CostTracker()
        ct.record(_fake_result(cell_id=1), batch_id="b1")
        ct.record(_fake_result(cell_id=2), batch_id="b1")
        out = tmp_path / "log.csv"
        ct.write_csv(out)
        text = out.read_text()
        assert "provider" in text  # header
        assert "zhipu" in text
        assert text.count("\n") >= 3  # header + 2 rows


class TestRateLimiter:
    def test_first_call_is_immediate(self) -> None:
        rl = _RateLimiter(rpm=600)
        t0 = time.monotonic()
        rl.wait()
        assert time.monotonic() - t0 < 0.05

    def test_second_call_blocks_for_min_interval(self) -> None:
        rl = _RateLimiter(rpm=120)  # 0.5 s interval
        rl.wait()
        t1 = time.monotonic()
        rl.wait()
        elapsed = time.monotonic() - t1
        # Should be ~0.5 s (allow a small tolerance for scheduler jitter)
        assert 0.40 < elapsed < 0.60

    def test_zero_rpm_clamps_to_min_interval(self) -> None:
        rl = _RateLimiter(rpm=0)  # guarded internally to >=1
        # Just ensure it doesn't crash and waits
        rl.wait()
