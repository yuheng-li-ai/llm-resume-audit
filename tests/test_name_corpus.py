"""Tests for NameCorpus (Phase 2.2).

Real-data integration tests skip if Census 2010 surname file is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_audit.name_corpus import (
    BLACK_THRESHOLD,
    DEFAULT_THRESHOLD,
    NameCorpus,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CENSUS_CSV = _REPO_ROOT / "data" / "raw" / "names" / "census2010" / "Names_2010Census.csv"


@pytest.fixture(scope="session")
def corpus() -> NameCorpus:
    if not _CENSUS_CSV.exists():
        pytest.skip(f"Census 2010 surnames not at {_CENSUS_CSV} (run Phase 2.1.1)")
    return NameCorpus(census_csv=_CENSUS_CSV)


class TestFirstNames:
    def test_returns_4_male_first_names(self, corpus: NameCorpus) -> None:
        names = corpus.first_names("male")
        assert len(names) == 4

    def test_returns_4_female_first_names(self, corpus: NameCorpus) -> None:
        names = corpus.first_names("female")
        assert len(names) == 4

    def test_male_first_name_gender_posteriors_at_least_0_99(self, corpus: NameCorpus) -> None:
        for rec in corpus.first_names("male"):
            assert rec.gender_posterior >= 0.99, f"{rec.first_name}: {rec.gender_posterior}"

    def test_female_first_name_gender_posteriors_at_least_0_99(self, corpus: NameCorpus) -> None:
        for rec in corpus.first_names("female"):
            assert rec.gender_posterior >= 0.99, f"{rec.first_name}: {rec.gender_posterior}"

    def test_first_names_are_distinct_within_gender(self, corpus: NameCorpus) -> None:
        male = {r.first_name for r in corpus.first_names("male")}
        female = {r.first_name for r in corpus.first_names("female")}
        assert len(male) == 4
        assert len(female) == 4

    def test_no_first_name_overlap_between_genders(self, corpus: NameCorpus) -> None:
        male = {r.first_name for r in corpus.first_names("male")}
        female = {r.first_name for r in corpus.first_names("female")}
        assert not (male & female)

    def test_unknown_gender_raises(self, corpus: NameCorpus) -> None:
        with pytest.raises(ValueError):
            corpus.first_names("nonbinary")  # type: ignore[arg-type]


class TestSurnames:
    @pytest.mark.parametrize("ethnicity", ["white", "hispanic", "asian"])
    def test_non_black_surnames_meet_default_threshold(
        self, corpus: NameCorpus, ethnicity: str
    ) -> None:
        rec = corpus.surname(ethnicity)
        assert (
            rec.posterior >= DEFAULT_THRESHOLD
        ), f"{rec.last_name} P({ethnicity})={rec.posterior} < {DEFAULT_THRESHOLD}"

    def test_black_surname_meets_relaxed_threshold(self, corpus: NameCorpus) -> None:
        rec = corpus.surname("black")
        assert rec.posterior >= BLACK_THRESHOLD

    def test_black_surname_is_washington(self, corpus: NameCorpus) -> None:
        rec = corpus.surname("black")
        assert rec.last_name == "Washington"

    def test_unknown_ethnicity_raises(self, corpus: NameCorpus) -> None:
        with pytest.raises(ValueError):
            corpus.surname("aian")  # type: ignore[arg-type]


class TestBuildCorpus:
    def test_returns_32_rows(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        assert len(df) == 32

    def test_has_required_columns(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        assert {
            "name_id",
            "first_name",
            "last_name",
            "gender",
            "ethnicity",
            "posterior_prob",
            "source",
        }.issubset(df.columns)

    def test_8_cells_with_4_rows_each(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        cell_sizes = df.groupby(["gender", "ethnicity"]).size()
        assert (cell_sizes == 4).all()
        assert len(cell_sizes) == 8

    def test_name_ids_unique(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        assert df["name_id"].nunique() == 32

    def test_white_hispanic_asian_posteriors_meet_default(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        non_black = df.loc[df["ethnicity"] != "black"]
        assert (non_black["posterior_prob"] >= DEFAULT_THRESHOLD).all()

    def test_black_posteriors_meet_relaxed(self, corpus: NameCorpus) -> None:
        df = corpus.build_corpus()
        black = df.loc[df["ethnicity"] == "black"]
        assert len(black) == 8
        assert (black["posterior_prob"] >= BLACK_THRESHOLD).all()


class TestPersistence:
    def test_round_trip_via_parquet(self, corpus: NameCorpus, tmp_path: Path) -> None:
        out = tmp_path / "name_corpus.parquet"
        corpus.write_corpus(out)
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert len(loaded) == 32
