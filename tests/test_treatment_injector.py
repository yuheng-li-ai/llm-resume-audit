"""Tests for TreatmentInjector (Phase 4.1).

Real-data integration tests skip if O*NET / name corpus are absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from llm_audit.name_corpus import NameCorpus
from llm_audit.onet_loader import OnetLoader
from llm_audit.resume_factory import ResumeFactory
from llm_audit.treatment_injector import TreatmentCell, TreatmentInjector
from llm_audit.utils.io import SeedConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONET_DIR = _REPO_ROOT / "data" / "raw" / "onet" / "db_28_1_text"
_CENSUS_CSV = _REPO_ROOT / "data" / "raw" / "names" / "census2010" / "Names_2010Census.csv"
_SEEDS_PATH = _REPO_ROOT / "config" / "seeds.toml"
_TEMPLATE_DIR = _REPO_ROOT / "templates"

KNOWN_SOC = "29-1141.00"  # Registered Nurses


@pytest.fixture(scope="session")
def injector() -> TreatmentInjector:
    if not (_ONET_DIR / "Occupation Data.txt").exists():
        pytest.skip(f"O*NET data not at {_ONET_DIR} (run Phase 1.1.1)")
    if not _CENSUS_CSV.exists():
        pytest.skip(f"Census 2010 not at {_CENSUS_CSV} (run Phase 2.1.1)")
    factory = ResumeFactory(
        onet_loader=OnetLoader(_ONET_DIR),
        seed_config=SeedConfig(_SEEDS_PATH),
        template_dir=_TEMPLATE_DIR,
    )
    name_corpus = NameCorpus(census_csv=_CENSUS_CSV).build_corpus()
    return TreatmentInjector(
        resume_factory=factory,
        name_corpus_df=name_corpus,
        seed_config=SeedConfig(_SEEDS_PATH),
    )


@pytest.fixture
def cell() -> TreatmentCell:
    return TreatmentCell(
        cell_id=0,
        resume_id=0,
        t_g="female",
        t_e="hispanic",
        t_p="mid_career",
        s_signal=True,
        occupation_soc=KNOWN_SOC,
        education_tier="bachelor",
    )


class TestInject:
    def test_returns_str(self, injector: TreatmentInjector, cell: TreatmentCell) -> None:
        body = injector.inject(cell)
        assert isinstance(body, str)
        assert len(body) > 200

    def test_no_template_placeholders_left(
        self, injector: TreatmentInjector, cell: TreatmentCell
    ) -> None:
        body = injector.inject(cell)
        assert "<<NAME>>" not in body
        assert "<<EMAIL>>" not in body
        assert "<<PHONE>>" not in body
        assert not re.search(r"<<[A-Z_]+>>", body)


class TestNameInjection:
    def test_first_name_matches_gender(
        self, injector: TreatmentInjector, cell: TreatmentCell
    ) -> None:
        body = injector.inject(cell)
        female_first = {"Mary", "Jennifer", "Patricia", "Linda"}
        first_line = body.split("\n", 1)[0].strip()
        first_name = first_line.split()[0]
        assert first_name in female_first

    def test_last_name_matches_ethnicity(
        self, injector: TreatmentInjector, cell: TreatmentCell
    ) -> None:
        body = injector.inject(cell)
        first_line = body.split("\n", 1)[0].strip()
        assert "Garcia" in first_line

    def test_male_white_lands_olson(self, injector: TreatmentInjector) -> None:
        c = TreatmentCell(
            cell_id=0,
            resume_id=1,
            t_g="male",
            t_e="white",
            t_p="early_career",
            s_signal=False,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        body = injector.inject(c)
        first_line = body.split("\n", 1)[0].strip()
        assert "Olson" in first_line


class TestAgeSignalConsistency:
    @pytest.mark.parametrize(
        "t_p, expected_years",
        [("early_career", 5), ("mid_career", 15), ("late_career", 25)],
    )
    def test_t_p_drives_total_years(
        self,
        injector: TreatmentInjector,
        t_p: str,
        expected_years: int,
    ) -> None:
        c = TreatmentCell(
            cell_id=0,
            resume_id=2,
            t_g="male",
            t_e="white",
            t_p=t_p,  # type: ignore[arg-type]
            s_signal=False,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        body = injector.inject(c)
        years_block = re.findall(r"\((\d{4})-\d{4}\)", body)
        assert years_block, "no (YYYY-YYYY) experience ranges found in body"
        earliest_start = min(int(y) for y in years_block)
        assert (
            2024 - earliest_start == expected_years
        ), f"t_p={t_p}: career_span={2024 - earliest_start} (expected {expected_years})"


class TestSignalToggle:
    def test_s_signal_true_includes_objective_block(self, injector: TreatmentInjector) -> None:
        c = TreatmentCell(
            cell_id=0,
            resume_id=3,
            t_g="male",
            t_e="black",
            t_p="mid_career",
            s_signal=True,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        body = injector.inject(c)
        assert "OBJECTIVE QUALIFICATIONS" in body
        assert "GPA:" in body

    def test_s_signal_false_omits_objective_block(self, injector: TreatmentInjector) -> None:
        c = TreatmentCell(
            cell_id=0,
            resume_id=3,
            t_g="male",
            t_e="black",
            t_p="mid_career",
            s_signal=False,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        body = injector.inject(c)
        assert "OBJECTIVE QUALIFICATIONS" not in body


class TestDeterminism:
    def test_same_cell_produces_same_body(
        self, injector: TreatmentInjector, cell: TreatmentCell
    ) -> None:
        a = injector.inject(cell)
        b = injector.inject(cell)
        assert a == b

    def test_different_resume_id_produces_different_body(self, injector: TreatmentInjector) -> None:
        c1 = TreatmentCell(
            cell_id=0,
            resume_id=10,
            t_g="female",
            t_e="asian",
            t_p="mid_career",
            s_signal=True,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        c2 = TreatmentCell(
            cell_id=0,
            resume_id=11,
            t_g="female",
            t_e="asian",
            t_p="mid_career",
            s_signal=True,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        assert injector.inject(c1) != injector.inject(c2)


class TestCellSchema:
    def test_treatment_cell_is_frozen(self, cell: TreatmentCell) -> None:
        with pytest.raises((AttributeError, TypeError)):
            cell.cell_id = 999  # type: ignore[misc]

    def test_treatment_cell_validates_t_g(self) -> None:
        with pytest.raises(ValueError):
            TreatmentCell(
                cell_id=0,
                resume_id=0,
                t_g="other",  # type: ignore[arg-type]
                t_e="white",
                t_p="mid_career",
                s_signal=False,
                occupation_soc=KNOWN_SOC,
                education_tier="bachelor",
            )

    def test_treatment_cell_validates_t_e(self) -> None:
        with pytest.raises(ValueError):
            TreatmentCell(
                cell_id=0,
                resume_id=0,
                t_g="male",
                t_e="aian",  # type: ignore[arg-type]
                t_p="mid_career",
                s_signal=False,
                occupation_soc=KNOWN_SOC,
                education_tier="bachelor",
            )


class TestBatch:
    def test_inject_batch_returns_dataframe(
        self, injector: TreatmentInjector, cell: TreatmentCell
    ) -> None:
        c2 = TreatmentCell(
            cell_id=1,
            resume_id=1,
            t_g="male",
            t_e="white",
            t_p="early_career",
            s_signal=False,
            occupation_soc=KNOWN_SOC,
            education_tier="bachelor",
        )
        df = injector.inject_batch([cell, c2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert {
            "cell_id",
            "resume_id",
            "t_g",
            "t_e",
            "t_p",
            "s_signal",
            "occupation_soc",
            "prompt_text",
        }.issubset(df.columns)
