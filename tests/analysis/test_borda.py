"""Phase 8.2 unit tests — Borda translation helpers."""

from __future__ import annotations

from llm_audit.analysis.borda import borda_score, borda_to_scale, parse_ranking


class TestParseRanking:
    def test_extracts_integers_in_order(self) -> None:
        text = "1. Candidate 42 is best. 2. Then 17. 3. Finally 99."
        out = parse_ranking(text, valid_ids={42, 17, 99, 5})
        assert out == [42, 17, 99]

    def test_filters_invalid_ids(self) -> None:
        text = "1) cell 1 2) cell 2 3) cell 999"
        out = parse_ranking(text, valid_ids={1, 2})
        assert out == [1, 2]

    def test_dedups_repeats(self) -> None:
        text = "42, 42, 17, 42"
        out = parse_ranking(text, valid_ids={42, 17})
        assert out == [42, 17]

    def test_empty_when_no_valid_ids_appear(self) -> None:
        out = parse_ranking("rank: alpha beta gamma", valid_ids={1, 2, 3})
        assert out == []


class TestBordaScore:
    def test_standard_5_candidate_borda(self) -> None:
        out = borda_score([10, 20, 30, 40, 50], top_score=5)
        assert out == {10: 5, 20: 4, 30: 3, 40: 2, 50: 1}

    def test_partial_ranking(self) -> None:
        out = borda_score([10, 20, 30], top_score=5)
        assert out == {10: 5, 20: 4, 30: 3}

    def test_default_top_score(self) -> None:
        out = borda_score([1, 2])
        assert max(out.values()) == 5


class TestBordaToScale:
    def test_top_rank_gets_target_max(self) -> None:
        borda = {10: 5, 20: 4, 30: 3, 40: 2, 50: 1}
        scaled = borda_to_scale(borda, group_size=5, target_max=100.0)
        assert scaled[10] == 100.0
        assert scaled[50] == 20.0

    def test_monotone_in_borda_points(self) -> None:
        borda = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        scaled = borda_to_scale(borda, group_size=5)
        ordered = [scaled[i] for i in range(1, 6)]
        assert ordered == sorted(ordered, reverse=True)
