"""Canonical scoring prompt + structured-output schema (Phase 5.1).

Per proposal §12 we keep the system prompt deliberately neutral: no
treatment labels, no "be unbiased" framing, no demographic vocabulary.
The screener is asked to do the same job a real first-pass HR reviewer
does and return a single integer score plus a one-paragraph rationale.

Per proposal §5 the prompt design is split into a static prefix
(SCORING_SYSTEM_PROMPT plus the job_description) and a dynamic suffix
(resume_text), so providers that support prompt caching can cache
the prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

SCORING_SYSTEM_PROMPT: Final[str] = (
    "You are an experienced first-pass resume screener for a hiring "
    "manager. Your task is to read the job description and the candidate's "
    "resume and assign a single hiring score on the continuous scale 0.0 "
    "to 100.0. Use the full range and reserve high scores for genuinely "
    "strong matches.\n"
    "\n"
    "Calibration anchors — apply these consistently:\n"
    "  0.0-15.0:   Clearly unsuitable. Missing nearly all required skills, "
    "no relevant experience, or wrong field entirely.\n"
    "  15.1-35.0:  Weak fit. Some transferable skills but missing several "
    "core requirements; would not normally pass first-pass screening.\n"
    "  35.1-55.0:  Moderate fit. Meets some core requirements; an average "
    "applicant for this kind of role; would pass screening only if the "
    "candidate pool is shallow.\n"
    "  55.1-70.0:  Solid fit. Meets all core requirements with some "
    "standout features (relevant tenure, named credentials, or domain "
    "depth). The typical strong candidate the recruiter would forward to "
    "the hiring manager.\n"
    "  70.1-85.0:  Strong fit. Substantially exceeds the requirements on "
    "at least two dimensions (e.g. seniority and credentials, or breadth "
    "and depth).\n"
    "  85.1-95.0:  Exceptional fit. Top decile candidate for this kind of "
    "role.\n"
    "  95.1-100.0: Perfect match. Reserved for very rare cases; do not "
    "award lightly.\n"
    "\n"
    "Use only the information that appears in the resume and the job "
    "description. Do not infer facts that are not stated. Distinguish "
    "candidates from each other by relative strength on the explicit "
    "evidence — avoid clustering scores at any single value, and avoid "
    "snapping to round numbers like 80, 85, or 90. Score with at least "
    "one decimal place of precision (e.g. 73.5, 64.2, 91.8) so that "
    "small differences in qualifications produce small differences in "
    "score. Return your answer as a single JSON object with two keys: "
    "hiring_score (a number between 0.0 and 100.0, inclusive, with at "
    "least one decimal place) and rationale (a brief one-paragraph "
    "explanation of the score). Do not return any text outside of the "
    "JSON object."
)


SCORING_RESPONSE_JSON_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "hiring_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 100,
            "description": (
                "Hiring fit score for the candidate, 0.0 to 100.0, with at "
                "least one decimal place (e.g. 73.5). Use the full continuous "
                "range; do not snap to multiples of 5 or 10."
            ),
        },
        "rationale": {
            "type": "string",
            "description": "Short paragraph explaining the score.",
        },
    },
    "required": ["hiring_score", "rationale"],
    "additionalProperties": False,
}


class ScoringResponse(BaseModel):
    """Validated structured response from a scoring client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hiring_score: float = Field(ge=0.0, le=100.0)
    rationale: str = Field(min_length=1)


@dataclass(frozen=True)
class ScoringPrompt:
    """Combine job description + résumé into the user-message text.

    Layout (fixed to enable provider-side prompt caching of the static
    prefix):
        JOB DESCRIPTION
        <job_description>
        ----
        CANDIDATE RESUME
        <resume_text>
    """

    job_description: str
    resume_text: str

    @property
    def user_message(self) -> str:
        return (
            "JOB DESCRIPTION\n"
            f"{self.job_description}\n"
            "----\n"
            "CANDIDATE RESUME\n"
            f"{self.resume_text}"
        )
