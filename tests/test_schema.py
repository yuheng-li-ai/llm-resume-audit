"""Tests for Pydantic Resume schema (Phase 1.2.1).

Resume is the structured source-of-truth; body_text strings are derived
via Jinja2 templating (Phase 1.2.2).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_audit.schema import (
    NAME_PLACEHOLDER,
    Certification,
    Contact,
    Education,
    Experience,
    ObjectiveSignals,
    Resume,
    ResumeMeta,
)


def _minimal_resume(**overrides: object) -> Resume:
    defaults: dict[str, object] = {
        "meta": ResumeMeta(
            resume_id=1,
            occupation_soc="15-1252.00",
            years_exp_bracket="mid_career",
            education_tier="bachelor",
        ),
        "contact": Contact(email="<<EMAIL>>", phone="<<PHONE>>"),
        "education": (
            Education(
                institution="State University",
                degree="B.S.",
                field="Computer Science",
                year=2014,
            ),
        ),
        "experience": (
            Experience(
                title="Software Engineer",
                employer="ACME Tech",
                start_year=2014,
                end_year=2024,
                bullets=("Built backend services.", "Led code reviews."),
            ),
        ),
        "skills": ("Python", "SQL", "Linux"),
    }
    defaults.update(overrides)
    return Resume(**defaults)


class TestResumeConstruction:
    def test_minimal_resume_constructs(self) -> None:
        r = _minimal_resume()
        assert r.name == NAME_PLACEHOLDER

    def test_default_name_is_placeholder(self) -> None:
        assert _minimal_resume().name == "<<NAME>>"

    def test_meta_required(self) -> None:
        with pytest.raises(ValidationError):
            Resume(  # type: ignore[call-arg]
                contact=Contact(email="x", phone="y"),
                education=(),
                experience=(),
                skills=(),
            )

    def test_certifications_default_empty(self) -> None:
        assert _minimal_resume().certifications == ()

    def test_objective_signals_default_none(self) -> None:
        assert _minimal_resume().objective_signals is None


class TestImmutability:
    def test_resume_is_frozen(self) -> None:
        r = _minimal_resume()
        with pytest.raises(ValidationError):
            r.name = "Alice Wonderland"  # type: ignore[misc]

    def test_education_is_frozen(self) -> None:
        e = Education(institution="MIT", degree="Ph.D.", field="EECS", year=2020)
        with pytest.raises(ValidationError):
            e.year = 2021  # type: ignore[misc]

    def test_education_collection_is_tuple(self) -> None:
        r = _minimal_resume()
        assert isinstance(r.education, tuple)


class TestEnumLikeFields:
    @pytest.mark.parametrize("bracket", ["early_career", "mid_career", "late_career"])
    def test_valid_years_exp_brackets(self, bracket: str) -> None:
        meta = ResumeMeta(
            resume_id=1,
            occupation_soc="15-1252.00",
            years_exp_bracket=bracket,
            education_tier="bachelor",
        )
        assert meta.years_exp_bracket == bracket

    def test_invalid_years_exp_bracket_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResumeMeta(
                resume_id=1,
                occupation_soc="15-1252.00",
                years_exp_bracket="senior_executive",  # type: ignore[arg-type]
                education_tier="bachelor",
            )

    @pytest.mark.parametrize(
        "tier", ["high_school", "associate", "bachelor", "master", "doctorate"]
    )
    def test_valid_education_tiers(self, tier: str) -> None:
        meta = ResumeMeta(
            resume_id=1,
            occupation_soc="15-1252.00",
            years_exp_bracket="mid_career",
            education_tier=tier,
        )
        assert meta.education_tier == tier

    def test_invalid_education_tier_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResumeMeta(
                resume_id=1,
                occupation_soc="15-1252.00",
                years_exp_bracket="mid_career",
                education_tier="postdoc",  # type: ignore[arg-type]
            )


class TestObjectiveSignals:
    def test_constructs_with_all_fields(self) -> None:
        sig = ObjectiveSignals(
            gpa=3.8,
            test_score_percentile=92.0,
            certification_ids=("PMP-12345", "AWS-CSA-67890"),
        )
        assert sig.gpa == 3.8
        assert sig.test_score_percentile == 92.0
        assert sig.certification_ids == ("PMP-12345", "AWS-CSA-67890")

    def test_gpa_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObjectiveSignals(gpa=5.0, test_score_percentile=50.0, certification_ids=())

    def test_percentile_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ObjectiveSignals(gpa=3.0, test_score_percentile=150.0, certification_ids=())


class TestSubModels:
    def test_certification_constructs(self) -> None:
        c = Certification(name="CPA", issuer="AICPA", year=2018)
        assert c.name == "CPA"

    def test_experience_bullets_is_tuple(self) -> None:
        e = Experience(
            title="Auditor",
            employer="Big Four",
            start_year=2018,
            end_year=2024,
            bullets=("Audited Fortune 500 clients.",),
        )
        assert isinstance(e.bullets, tuple)
