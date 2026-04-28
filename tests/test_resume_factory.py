"""Tests for ResumeFactory (Phase 1.2.3).

Real-data integration tests skip if O*NET 28.1 bundle is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_audit.onet_loader import OnetLoader
from llm_audit.resume_factory import ResumeFactory
from llm_audit.schema import NAME_PLACEHOLDER, Resume
from llm_audit.utils.io import OccupationsConfig, SeedConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ONET_DIR = _REPO_ROOT / "data" / "raw" / "onet" / "db_28_1_text"
_SEEDS_PATH = _REPO_ROOT / "config" / "seeds.toml"
_OCCUPATIONS_PATH = _REPO_ROOT / "config" / "occupations.toml"
_TEMPLATE_DIR = _REPO_ROOT / "templates"

KNOWN_SOC = "15-1252.00"


@pytest.fixture(scope="session")
def factory() -> ResumeFactory:
    if not (_ONET_DIR / "Occupation Data.txt").exists():
        pytest.skip(f"O*NET data not at {_ONET_DIR} (run Phase 1.1.1 download)")
    return ResumeFactory(
        onet_loader=OnetLoader(_ONET_DIR),
        seed_config=SeedConfig(_SEEDS_PATH),
        template_dir=_TEMPLATE_DIR,
    )


@pytest.fixture(scope="session")
def all_18_socs() -> tuple[str, ...]:
    occs = OccupationsConfig(_OCCUPATIONS_PATH).occupations
    return tuple(o["onet_soc"] for o in occs)


class TestBuildOne:
    def test_returns_resume_instance(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=0,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert isinstance(r, Resume)

    def test_meta_fields_propagate(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=42,
            soc=KNOWN_SOC,
            years_exp_bracket="early_career",
            education_tier="master",
        )
        assert r.meta.resume_id == 42
        assert r.meta.occupation_soc == KNOWN_SOC
        assert r.meta.years_exp_bracket == "early_career"
        assert r.meta.education_tier == "master"

    def test_name_is_placeholder(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert r.name == NAME_PLACEHOLDER

    def test_contact_uses_placeholders(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert "<<EMAIL>>" in r.contact.email
        assert "<<PHONE>>" in r.contact.phone

    def test_has_at_least_one_education_and_experience(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert len(r.education) >= 1
        assert len(r.experience) >= 1

    def test_with_signals_attaches_objective_signals(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
            with_signals=True,
        )
        assert r.objective_signals is not None
        assert 0.0 <= r.objective_signals.gpa <= 4.0

    def test_unknown_soc_raises(self, factory: ResumeFactory) -> None:
        with pytest.raises(ValueError, match="No O\\*NET tasks for"):
            factory.build_one(
                resume_id=1,
                soc="99-9999.99",
                years_exp_bracket="mid_career",
                education_tier="bachelor",
            )


class TestDeterminism:
    def test_same_seed_and_id_yields_identical_resume(self, factory: ResumeFactory) -> None:
        a = factory.build_one(
            resume_id=7,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        b = factory.build_one(
            resume_id=7,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert a == b

    def test_different_id_yields_different_resume(self, factory: ResumeFactory) -> None:
        a = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        b = factory.build_one(
            resume_id=2,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        assert a != b


class TestRender:
    def test_render_returns_text(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        body = factory.render(r)
        assert isinstance(body, str)
        assert NAME_PLACEHOLDER in body
        assert "EDUCATION" in body
        assert "EXPERIENCE" in body
        assert "SKILLS" in body

    def test_no_llm_stamp_phrases(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        body = factory.render(r)
        for forbidden in ("As an AI", "In conclusion", "I am an AI", "language model"):
            assert forbidden not in body, f"LLM-stamp phrase {forbidden!r} leaked into body"

    def test_body_within_token_budget(self, factory: ResumeFactory) -> None:
        r = factory.build_one(
            resume_id=1,
            soc=KNOWN_SOC,
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        )
        body = factory.render(r)
        assert 500 <= len(body) <= 3500, f"body length {len(body)} outside [500, 3500]"


class TestBuildPanel:
    def test_build_panel_returns_n_per_occupation_times_18(
        self, factory: ResumeFactory, all_18_socs: tuple[str, ...]
    ) -> None:
        panel = factory.build_panel(occupations=all_18_socs, n_per_occupation=2)
        assert len(panel) == 2 * len(all_18_socs)

    def test_build_panel_resume_ids_unique(
        self, factory: ResumeFactory, all_18_socs: tuple[str, ...]
    ) -> None:
        panel = factory.build_panel(occupations=all_18_socs, n_per_occupation=2)
        ids = [r.meta.resume_id for r in panel]
        assert len(set(ids)) == len(ids)

    def test_build_panel_covers_all_18_socs(
        self, factory: ResumeFactory, all_18_socs: tuple[str, ...]
    ) -> None:
        panel = factory.build_panel(occupations=all_18_socs, n_per_occupation=1)
        socs_seen = {r.meta.occupation_soc for r in panel}
        assert socs_seen == set(all_18_socs)


class TestSnapshot:
    """Phase 1.3: byte-for-byte snapshot of the locked-seed panel.

    Any deliberate change to ResumeFactory or templates/resume.j2 must
    consciously regenerate the fixtures via:
        python -m llm_audit.resume_factory  (or the inline driver)
    """

    @pytest.fixture(scope="class")
    def panel(self, factory: ResumeFactory, all_18_socs: tuple[str, ...]) -> tuple[Resume, ...]:
        return factory.build_panel(occupations=all_18_socs, n_per_occupation=25)

    def test_panel_size_locked_at_450(self, panel: tuple[Resume, ...]) -> None:
        assert len(panel) == 450

    @pytest.mark.parametrize("idx", [0, 449])
    def test_snapshot_matches(
        self, factory: ResumeFactory, panel: tuple[Resume, ...], idx: int
    ) -> None:
        fixture_path = _REPO_ROOT / "tests" / "fixtures" / f"resume_{idx:03d}.txt"
        expected = fixture_path.read_text()
        actual = factory.render(panel[idx])
        assert actual == expected, (
            f"Résumé #{idx} drifted from snapshot at {fixture_path}. "
            f"If this change is intentional, regenerate the fixture."
        )
