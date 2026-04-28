"""Tests for JobDescriptions (Phase 3.1-3.2).

Real-data integration tests skip if O*NET 28.1 bundle is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_audit.job_descriptions import (
    BANNED_DEMOGRAPHIC_WORDS,
    JobDescriptions,
)
from llm_audit.onet_loader import OnetLoader
from llm_audit.utils.io import OccupationsConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONET_DIR = _REPO_ROOT / "data" / "raw" / "onet" / "db_28_1_text"
_OCCUPATIONS_PATH = _REPO_ROOT / "config" / "occupations.toml"


@pytest.fixture(scope="session")
def jobs() -> JobDescriptions:
    if not (_ONET_DIR / "Occupation Data.txt").exists():
        pytest.skip(f"O*NET data not at {_ONET_DIR} (run Phase 1.1.1)")
    return JobDescriptions(
        onet_loader=OnetLoader(_ONET_DIR),
        occupations_config=OccupationsConfig(_OCCUPATIONS_PATH),
    )


class TestBuild:
    def test_returns_54_rows(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert len(df) == 54

    def test_required_columns(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert {"occupation_soc", "phrasing_id", "title", "summary", "requirements"}.issubset(
            df.columns
        )

    def test_three_phrasings_per_occupation(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        per_soc = df.groupby("occupation_soc").size()
        assert (per_soc == 3).all()
        assert len(per_soc) == 18

    def test_phrasing_ids_are_0_1_2(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert sorted(df["phrasing_id"].unique().tolist()) == [0, 1, 2]

    def test_title_non_empty(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert (df["title"].str.len() > 0).all()

    def test_summary_non_empty(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert (df["summary"].str.len() > 50).all()

    def test_requirements_non_empty(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        assert (df["requirements"].str.len() > 0).all()

    def test_no_phrasing_has_empty_skill_list(self, jobs: JobDescriptions) -> None:
        # Phrasing 0 = "Required skills: <skills>." — must contain at least one
        # skill name (not just "Required skills: ."). Catches the Tier-2 SOC
        # bug that motivated _PROXY_FOR_RATINGS.
        df = jobs.build()
        phrasing0 = df.loc[df["phrasing_id"] == 0]
        for _, row in phrasing0.iterrows():
            assert (
                row["requirements"] != "Required skills: ."
            ), f"Empty skill list for SOC {row['occupation_soc']} — proxy lookup failed"

    def test_proxy_socs_resolve_to_non_empty_skills(self, jobs: JobDescriptions) -> None:
        # The two Tier-2 SOCs must come back with full skill lists via proxy.
        df = jobs.build()
        for tier2_soc in ("15-1252.00", "13-2051.00"):
            row = df.loc[(df["occupation_soc"] == tier2_soc) & (df["phrasing_id"] == 0)]
            assert len(row) == 1
            req = row.iloc[0]["requirements"]
            assert "Required skills:" in req
            assert (
                len(req) > len("Required skills: .") + 5
            ), f"{tier2_soc} got an empty proxy lookup: {req!r}"


class TestStopWordLint:
    def test_no_banned_demographic_words_in_output(self, jobs: JobDescriptions) -> None:
        df = jobs.build()
        violations = jobs.lint(df)
        assert violations == [], f"Demographic stop-words leaked: {violations}"

    def test_lint_detects_seeded_violation_with_full_provenance(
        self, jobs: JobDescriptions
    ) -> None:
        df = jobs.build().copy()
        target_soc = df.iloc[0]["occupation_soc"]
        df.loc[0, "summary"] = df.loc[0, "summary"] + " Looking for a young energetic candidate."
        violations = jobs.lint(df)
        assert violations, "Lint must detect seeded violation"
        msg = violations[0]
        assert target_soc in msg
        assert "summary" in msg
        assert "young" in msg.lower() or "energetic" in msg.lower()
        assert "phrasing" in msg.lower()

    def test_banned_list_non_trivial(self) -> None:
        # Guard against an empty banned list silently passing every input
        assert len(BANNED_DEMOGRAPHIC_WORDS) >= 10


class TestPersistenceBlocksOnViolation:
    def test_write_succeeds_when_clean(self, jobs: JobDescriptions, tmp_path: Path) -> None:
        out = tmp_path / "job_descriptions.parquet"
        jobs.write(out)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 54

    def test_write_raises_when_violation_present(
        self, jobs: JobDescriptions, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force a violation by monkeypatching build to inject a banned word
        original_build = jobs.build

        def poisoned_build() -> pd.DataFrame:
            df = original_build().copy()
            df.loc[0, "summary"] = "Hiring a young energetic candidate."
            return df

        monkeypatch.setattr(jobs, "build", poisoned_build)
        out = tmp_path / "job_descriptions.parquet"
        with pytest.raises(ValueError, match="demographic"):
            jobs.write(out)
        assert not out.exists(), "Failed builds must not leave stale parquet"
