"""Pydantic v2 schema for the structured Resume source-of-truth.

`body_text` (the LLM-facing prose) is derived from `Resume` via the Jinja2
template in `templates/resume.j2`; this module owns only the structured
representation and its invariants.

All models are frozen and use tuples for collection fields, giving us
deep immutability per CLAUDE rule "Immutability (CRITICAL)".
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

NAME_PLACEHOLDER: Final[str] = "<<NAME>>"
EMAIL_PLACEHOLDER: Final[str] = "<<EMAIL>>"
PHONE_PLACEHOLDER: Final[str] = "<<PHONE>>"

YearsExpBracket = Literal["early_career", "mid_career", "late_career"]
EducationTier = Literal["high_school", "associate", "bachelor", "master", "doctorate"]


class _Frozen(BaseModel):
    """Common config: frozen + strict assignment validation."""

    model_config = ConfigDict(frozen=True, validate_assignment=True, extra="forbid")


class ResumeMeta(_Frozen):
    resume_id: int = Field(ge=0)
    occupation_soc: str
    years_exp_bracket: YearsExpBracket
    education_tier: EducationTier


class Contact(_Frozen):
    email: str
    phone: str


class Education(_Frozen):
    institution: str
    degree: str
    field: str
    year: int = Field(ge=1900, le=2100)


class Experience(_Frozen):
    title: str
    employer: str
    start_year: int = Field(ge=1900, le=2100)
    end_year: int = Field(ge=1900, le=2100)
    bullets: tuple[str, ...]


class Certification(_Frozen):
    name: str
    issuer: str
    year: int = Field(ge=1900, le=2100)


class ObjectiveSignals(_Frozen):
    gpa: float = Field(ge=0.0, le=4.0)
    test_score_percentile: float = Field(ge=0.0, le=100.0)
    certification_ids: tuple[str, ...]


class Resume(_Frozen):
    meta: ResumeMeta
    name: str = NAME_PLACEHOLDER
    contact: Contact
    summary: str = ""
    education: tuple[Education, ...]
    experience: tuple[Experience, ...]
    skills: tuple[str, ...]
    certifications: tuple[Certification, ...] = ()
    objective_signals: ObjectiveSignals | None = None
