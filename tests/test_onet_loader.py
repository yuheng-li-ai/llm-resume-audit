"""Tests for the OnetLoader class.

Real-data integration tests: skip when O*NET 28.1 bundle is not present
(Phase 1.1.1 download step). Construction-error tests run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_audit.onet_loader import OnetLoader

ONET_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "onet" / "db_28_1_text"
KNOWN_SOC = "15-1252.00"
KNOWN_TITLE = "Software Developers"


@pytest.fixture(scope="session")
def onet_dir() -> Path:
    if not (ONET_DIR / "Occupation Data.txt").exists():
        pytest.skip(f"O*NET data not present at {ONET_DIR} (run Phase 1.1.1 download)")
    return ONET_DIR


class TestConstruction:
    def test_constructs_with_existing_dir(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        assert loader.data_dir == onet_dir

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            OnetLoader(tmp_path / "nonexistent")

    def test_raises_on_dir_missing_required_files(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_onet"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="Occupation Data.txt"):
            OnetLoader(empty)


class TestLoadOccupations:
    def test_returns_dataframe_with_canonical_columns(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_occupations()
        assert isinstance(df, pd.DataFrame)
        assert {"onet_soc", "title", "description"}.issubset(df.columns)

    def test_at_least_1000_occupations(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        assert len(loader.load_occupations()) >= 1000

    def test_known_soc_present_with_expected_title(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_occupations()
        rows = df.loc[df["onet_soc"] == KNOWN_SOC]
        assert len(rows) == 1
        assert rows.iloc[0]["title"] == KNOWN_TITLE

    def test_caches_subsequent_calls_to_same_frame(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        first = loader.load_occupations()
        second = loader.load_occupations()
        assert first is second


class TestLoadTasks:
    def test_returns_dataframe_with_canonical_columns(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_tasks()
        assert {"onet_soc", "task_id", "task", "task_type"}.issubset(df.columns)

    def test_at_least_18000_tasks(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        assert len(loader.load_tasks()) >= 18000


class TestLoadTasksForSoc:
    def test_returns_non_empty_subset_for_known_soc(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_tasks_for_soc(KNOWN_SOC)
        assert len(df) > 0
        assert (df["onet_soc"] == KNOWN_SOC).all()

    def test_returns_empty_for_unknown_soc(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_tasks_for_soc("99-9999.99")
        assert len(df) == 0


class TestLoadSkillsAbilitiesKnowledge:
    def test_load_skills_returns_dataframe(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_skills()
        assert {
            "onet_soc",
            "element_id",
            "element_name",
            "scale_id",
            "data_value",
        }.issubset(df.columns)
        assert len(df) > 0

    def test_load_abilities_returns_dataframe(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_abilities()
        assert {"onet_soc", "element_id", "element_name"}.issubset(df.columns)
        assert len(df) > 0

    def test_load_knowledge_returns_dataframe(self, onet_dir: Path) -> None:
        loader = OnetLoader(onet_dir)
        df = loader.load_knowledge()
        assert {"onet_soc", "element_id", "element_name"}.issubset(df.columns)
        assert len(df) > 0
