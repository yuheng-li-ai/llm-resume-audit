"""Phase 1.1.3 verification: every SOC in config/occupations.toml resolves
to a non-empty row in O*NET 28.1 Occupation Data.

Acceptance criterion for the locked 3 x 3 x 2 occupation panel
(see CHECKLIST rule 4 and config/occupations.toml).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llm_audit.onet_loader import OnetLoader
from llm_audit.utils.io import OccupationsConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / "config" / "occupations.toml"
_ONET_DIR = _REPO_ROOT / "data" / "raw" / "onet" / "db_28_1_text"


@pytest.fixture(scope="session")
def occupations() -> tuple[dict[str, Any], ...]:
    return OccupationsConfig(_CONFIG_PATH).occupations


@pytest.fixture(scope="session")
def onet_loader() -> OnetLoader:
    if not (_ONET_DIR / "Occupation Data.txt").exists():
        pytest.skip(f"O*NET data not at {_ONET_DIR} (run Phase 1.1.1 download)")
    return OnetLoader(_ONET_DIR)


class TestPanelStructure:
    def test_exactly_18_occupations(self, occupations: tuple[dict[str, Any], ...]) -> None:
        assert len(occupations) == 18

    def test_balanced_3x3x2_panel(self, occupations: tuple[dict[str, Any], ...]) -> None:
        cells: dict[tuple[str, str], int] = {}
        for o in occupations:
            key = (o["stereotype"], o["skill_tier"])
            cells[key] = cells.get(key, 0) + 1
        for stereo in ("male", "female", "neutral"):
            for tier in ("high", "mid", "low"):
                assert (
                    cells.get((stereo, tier)) == 2
                ), f"Cell ({stereo}, {tier}) has {cells.get((stereo, tier), 0)} occupations, expected 2"

    def test_all_soc_codes_unique(self, occupations: tuple[dict[str, Any], ...]) -> None:
        socs = [o["soc"] for o in occupations]
        assert len(set(socs)) == len(socs)

    def test_all_onet_soc_codes_unique(self, occupations: tuple[dict[str, Any], ...]) -> None:
        codes = [o["onet_soc"] for o in occupations]
        assert len(set(codes)) == len(codes)


class TestOnetResolution:
    def test_every_onet_soc_resolves_in_onet(
        self,
        occupations: tuple[dict[str, Any], ...],
        onet_loader: OnetLoader,
    ) -> None:
        df = onet_loader.load_occupations()
        onet_codes = set(df["onet_soc"])
        missing = [o["onet_soc"] for o in occupations if o["onet_soc"] not in onet_codes]
        assert not missing, f"O*NET-SOC codes not in O*NET 28.1: {missing}"

    def test_titles_match_onet_canonical(
        self,
        occupations: tuple[dict[str, Any], ...],
        onet_loader: OnetLoader,
    ) -> None:
        df = onet_loader.load_occupations()
        title_by_code = dict(zip(df["onet_soc"], df["title"], strict=True))
        mismatches: list[tuple[str, str, str]] = []
        for o in occupations:
            canonical = title_by_code.get(o["onet_soc"])
            if canonical != o["title"]:
                mismatches.append((o["onet_soc"], o["title"], str(canonical)))
        assert not mismatches, f"Title mismatches (soc, config, onet): {mismatches}"

    def test_every_soc_has_at_least_one_task(
        self,
        occupations: tuple[dict[str, Any], ...],
        onet_loader: OnetLoader,
    ) -> None:
        empty: list[str] = []
        for o in occupations:
            tasks = onet_loader.load_tasks_for_soc(o["onet_soc"])
            if len(tasks) == 0:
                empty.append(o["onet_soc"])
        assert not empty, f"O*NET-SOC codes with zero task statements: {empty}"
