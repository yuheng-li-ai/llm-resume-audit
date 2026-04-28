"""TreatmentInjector — combine a base résumé with a (gender, ethnicity,
age-signal, objective-signal) treatment cell and produce the prompt_text
that an LLM screener will see.

Pipeline:
    1. Re-run ResumeFactory.build_one with t_p as years_exp_bracket and
       s_signal as with_signals. This guarantees Education year, Experience
       start/end years, and the OBJECTIVE QUALIFICATIONS block are all
       internally consistent with the assigned treatment.
    2. Render to body text via the same Jinja2 template.
    3. Pick (first_name, last_name) deterministically from name_corpus_df
       by (t_g, t_e), seeded by (resume_id, cell_id) so each résumé-treatment
       pair gets a stable choice.
    4. Substitute <<NAME>> -> "First Last", <<EMAIL>> -> "first.last@example.com",
       <<PHONE>> -> a deterministic 10-digit US-format number.

ZERO LLM API calls anywhere in this module — see proposal §5.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Final, Literal

import pandas as pd

from llm_audit.resume_factory import ResumeFactory
from llm_audit.schema import EducationTier, YearsExpBracket
from llm_audit.utils.io import SeedConfig

Gender = Literal["male", "female"]
Ethnicity = Literal["white", "black", "asian", "hispanic"]

_VALID_GENDERS: Final[frozenset[str]] = frozenset({"male", "female"})
_VALID_ETHNICITIES: Final[frozenset[str]] = frozenset({"white", "black", "asian", "hispanic"})
_VALID_BRACKETS: Final[frozenset[str]] = frozenset({"early_career", "mid_career", "late_career"})

_NAME_PLACEHOLDER: Final[str] = "<<NAME>>"
_EMAIL_PLACEHOLDER: Final[str] = "<<EMAIL>>"
_PHONE_PLACEHOLDER: Final[str] = "<<PHONE>>"


@dataclass(frozen=True)
class TreatmentCell:
    cell_id: int
    resume_id: int
    t_g: Gender
    t_e: Ethnicity
    t_p: YearsExpBracket
    s_signal: bool
    occupation_soc: str
    education_tier: EducationTier

    def __post_init__(self) -> None:
        if self.t_g not in _VALID_GENDERS:
            raise ValueError(f"Unknown t_g: {self.t_g!r}")
        if self.t_e not in _VALID_ETHNICITIES:
            raise ValueError(f"Unknown t_e: {self.t_e!r}")
        if self.t_p not in _VALID_BRACKETS:
            raise ValueError(f"Unknown t_p: {self.t_p!r}")

    def _replace(self, **changes: object) -> TreatmentCell:
        return replace(self, **changes)  # type: ignore[arg-type]


class TreatmentInjector:
    """Render a fully-injected résumé prompt for a TreatmentCell."""

    def __init__(
        self,
        resume_factory: ResumeFactory,
        name_corpus_df: pd.DataFrame,
        seed_config: SeedConfig,
    ) -> None:
        required = {"first_name", "last_name", "gender", "ethnicity"}
        if not required.issubset(name_corpus_df.columns):
            raise ValueError(
                f"name_corpus_df missing required columns: "
                f"{required - set(name_corpus_df.columns)}"
            )
        self._factory = resume_factory
        self._name_corpus = name_corpus_df.reset_index(drop=True)
        self._master_seed = seed_config.get("treatment_injector", "master")

    def inject(self, cell: TreatmentCell) -> str:
        resume = self._factory.build_one(
            resume_id=cell.resume_id,
            soc=cell.occupation_soc,
            years_exp_bracket=cell.t_p,
            education_tier=cell.education_tier,
            with_signals=cell.s_signal,
        )
        body = self._factory.render(resume)
        first, last = self._pick_name(cell)
        email, phone = self._pick_contact(cell, first, last)
        body = body.replace(_NAME_PLACEHOLDER, f"{first} {last}")
        body = body.replace(_EMAIL_PLACEHOLDER, email)
        body = body.replace(_PHONE_PLACEHOLDER, phone)
        return body

    def inject_batch(self, cells: Iterable[TreatmentCell]) -> pd.DataFrame:
        rows = [
            {
                "cell_id": c.cell_id,
                "resume_id": c.resume_id,
                "t_g": c.t_g,
                "t_e": c.t_e,
                "t_p": c.t_p,
                "s_signal": c.s_signal,
                "occupation_soc": c.occupation_soc,
                "prompt_text": self.inject(c),
            }
            for c in cells
        ]
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- private

    def _pick_name(self, cell: TreatmentCell) -> tuple[str, str]:
        sub = self._name_corpus.loc[
            (self._name_corpus["gender"] == cell.t_g) & (self._name_corpus["ethnicity"] == cell.t_e)
        ]
        if sub.empty:
            raise ValueError(f"No names in corpus for ({cell.t_g}, {cell.t_e})")
        rng = random.Random(self._master_seed + cell.resume_id * 13 + cell.cell_id)
        idx = rng.randrange(len(sub))
        row = sub.iloc[idx]
        return str(row["first_name"]), str(row["last_name"])

    def _pick_contact(
        self,
        cell: TreatmentCell,
        first: str,
        last: str,
    ) -> tuple[str, str]:
        rng = random.Random(self._master_seed + cell.resume_id * 7 + cell.cell_id * 3)
        email = f"{first.lower()}.{last.lower()}@example.com"
        area = rng.randint(200, 989)
        prefix = rng.randint(200, 989)
        line = rng.randint(1000, 9999)
        phone = f"({area}) {prefix:03d}-{line:04d}"
        return email, phone
