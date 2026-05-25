"""Phase 8.2 — Borda-count score-format robustness (proposal §7-2).

Helpers used by scripts/run_phase8_borda.py:

  parse_ranking(text, valid_ids)  -> list[int]
      Parse an LLM ranking response (free-form text) into an ordered
      list of cell_ids restricted to the given valid_ids set.

  borda_score(ranking, top_score)  -> dict[int, int]
      Convert a ranked list of cell_ids into per-cell Borda points
      (1st = top_score, 2nd = top_score - 1, ..., last = top_score - n + 1).

  borda_to_scale(borda_points, group_size, target_max=100.0) -> dict[int, float]
      Normalise per-cell Borda points to the same 0-100 scale used by the
      direct-scoring elicitation, so the comparison OLS coefficients are
      on comparable units.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


def parse_ranking(text: str, valid_ids: Iterable[int]) -> list[int]:
    """Extract ordered cell_id integers from a free-form ranking response.

    Strategy: scan the text for integers, keep only those in valid_ids, and
    preserve the order of first appearance. If the response repeats an id,
    only the first occurrence is kept.
    """
    valid = set(valid_ids)
    seen: set[int] = set()
    out: list[int] = []
    for token in re.findall(r"-?\d+", text):
        i = int(token)
        if i in valid and i not in seen:
            out.append(i)
            seen.add(i)
    return out


def borda_score(ranking: list[int], top_score: int = 5) -> dict[int, int]:
    """Convert a ranked list of cell_ids to per-cell Borda points.

    Standard Borda: rank 1 -> top_score points, rank 2 -> top_score - 1,
    ..., rank n -> top_score - n + 1. The lowest-ranked candidate gets
    `top_score - len(ranking) + 1` points (can be zero or negative if
    top_score < len(ranking); callers should set top_score >= len(ranking)).
    """
    return {cid: top_score - i for i, cid in enumerate(ranking)}


def borda_to_scale(
    borda_points: dict[int, int],
    group_size: int,
    target_max: float = 100.0,
) -> dict[int, float]:
    """Linearly scale Borda points in {1, ..., group_size} to (0, target_max].

    Rank 1 maps to target_max; rank `group_size` maps to target_max / group_size.
    """
    return {cid: float(p) / float(group_size) * target_max for cid, p in borda_points.items()}
