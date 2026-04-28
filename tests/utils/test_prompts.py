"""Tests for the canonical scoring prompt + JSON schema (Phase 5.1)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llm_audit.utils.prompts import (
    SCORING_RESPONSE_JSON_SCHEMA,
    SCORING_SYSTEM_PROMPT,
    ScoringPrompt,
    ScoringResponse,
)


class TestSystemPrompt:
    def test_system_prompt_is_non_empty(self) -> None:
        assert len(SCORING_SYSTEM_PROMPT) > 100

    def test_system_prompt_has_no_treatment_labels(self) -> None:
        # Per proposal §12: system prompt must not name treatments.
        # Whole-word match — "manager" contains "age" but that's not a leak.
        import re

        forbidden = (
            "gender",
            "ethnicity",
            "race",
            "demographic",
            r"\bage\b",
            r"\bsex\b",
            "discriminat",
            "stereotyp",
            "minority",
        )
        text = SCORING_SYSTEM_PROMPT.lower()
        for pattern in forbidden:
            assert not re.search(
                pattern, text
            ), f"System prompt leaks treatment label matching {pattern!r}"

    def test_system_prompt_specifies_json_output(self) -> None:
        assert "JSON" in SCORING_SYSTEM_PROMPT or "json" in SCORING_SYSTEM_PROMPT
        assert "0" in SCORING_SYSTEM_PROMPT and "100" in SCORING_SYSTEM_PROMPT


class TestScoringPrompt:
    def test_constructs_with_job_and_resume(self) -> None:
        p = ScoringPrompt(
            job_description="Software developer position requiring Python.",
            resume_text="James Olson\nB.S. Computer Science\n...",
        )
        assert isinstance(p.user_message, str)
        assert "Software developer" in p.user_message
        assert "James Olson" in p.user_message

    def test_user_message_does_not_contain_treatment_labels(self) -> None:
        p = ScoringPrompt(
            job_description="Position description",
            resume_text="Mary Garcia, B.S. Nursing",
        )
        forbidden = ("treatment", "t_g", "t_e", "t_p", "s_signal", "do(")
        for word in forbidden:
            assert word not in p.user_message.lower()

    def test_total_length_is_reasonable(self) -> None:
        p = ScoringPrompt(
            job_description="x" * 500,
            resume_text="y" * 1700,
        )
        total = len(SCORING_SYSTEM_PROMPT) + len(p.user_message)
        assert 1500 < total < 4000


class TestScoringResponse:
    def test_valid_construction(self) -> None:
        r = ScoringResponse(hiring_score=75, rationale="Strong fit on skills.")
        assert r.hiring_score == 75

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoringResponse(hiring_score=-1, rationale="x")

    def test_score_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoringResponse(hiring_score=101, rationale="x")

    def test_rationale_required(self) -> None:
        with pytest.raises(ValidationError):
            ScoringResponse(hiring_score=50)  # type: ignore[call-arg]

    def test_round_trip_json(self) -> None:
        r = ScoringResponse(hiring_score=42, rationale="Ok candidate.")
        payload = r.model_dump_json()
        loaded = ScoringResponse.model_validate_json(payload)
        assert loaded == r


class TestJsonSchema:
    def test_schema_is_valid_json_serializable(self) -> None:
        s = json.dumps(SCORING_RESPONSE_JSON_SCHEMA)
        assert "hiring_score" in s
        assert "rationale" in s

    def test_schema_declares_required_fields(self) -> None:
        assert "required" in SCORING_RESPONSE_JSON_SCHEMA
        required = set(SCORING_RESPONSE_JSON_SCHEMA["required"])
        assert {"hiring_score", "rationale"}.issubset(required)

    def test_schema_score_range_0_100(self) -> None:
        score_prop = SCORING_RESPONSE_JSON_SCHEMA["properties"]["hiring_score"]
        assert score_prop["type"] == "integer"
        assert score_prop["minimum"] == 0
        assert score_prop["maximum"] == 100
