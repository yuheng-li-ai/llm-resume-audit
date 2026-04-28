"""JobDescriptions — three mechanical-templating phrasings per occupation.

Per proposal §5 + Phase 3 design call (2026-04-28):
  - All variable content sourced from O*NET 28.1 canonical fields
    (Title, Description, Skills, Knowledge).
  - All fixed wording is hand-authored template literals in this module.
  - No LLM API call is made anywhere in the build path; the audit prompts
    must contain zero LLM-generated prose so the screener cannot
    pattern-match its own writing style.

Output: 54 rows (18 occupations x 3 phrasings) with schema
[occupation_soc, phrasing_id, title, summary, requirements].

Demographic-signal lint (BLOCKING): if any banned dog-whistle term
appears in title / summary / requirements, write() raises ValueError
with the offending occupation, phrasing, field, and token reported.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import pandas as pd

from llm_audit.onet_loader import OnetLoader
from llm_audit.utils.io import OccupationsConfig

# Conservative dog-whistle list. Each entry is matched as a whole-word
# (or whole-phrase) regex against title + summary + requirements.
BANNED_DEMOGRAPHIC_WORDS: Final[tuple[str, ...]] = (
    "young",
    "energetic",
    "vibrant",
    "fresh graduate",
    "recent graduate",
    "recent grad",
    "nurturing",
    "attractive",
    "native speaker",
    "native english",
    "native english speaker",
    "culture fit",
    "cultural fit",
    "wholesome",
    "physically fit",
    "athletic build",
)

_TOP_K_SKILLS: Final[int] = 6
_TOP_K_KNOWLEDGE: Final[int] = 4

# Skills/Knowledge proxy mapping for O*NET 28.1 Tier-2 occupations.
#
# Two of the 18 locked occupations appear in O*NET 28.1 `Occupation Data.txt`
# (Title and Description present) but have no rows in `Skills.txt` or
# `Knowledge.txt` because the O*NET research team has not yet collected
# importance/level ratings for them. For those two SOCs we read Skills and
# Knowledge from the closest Tier-1 sibling SOC; the audit-facing Title and
# Description remain the locked-SOC canonical text.
#
#   15-1252.00 Software Developers
#       -> 15-1251.00 Computer Programmers (closest semantic match in
#          O*NET 28.1 Skills.txt; supersedes the pre-2018 SOC 15-1132.00
#          "Software Developers, Applications" lineage).
#
#   13-2051.00 Financial and Investment Analysts
#       -> 13-2099.01 Financial Quantitative Analysts (closest semantic
#          match; both belong to SOC minor group 13-20 Financial
#          Specialists).
#
# Phase 3 design call (2026-04-28); the 18-occupation panel itself stays
# locked per CHECKLIST rule 4. Documented in proposal §5 data note.
_PROXY_FOR_RATINGS: Final[dict[str, str]] = {
    "15-1252.00": "15-1251.00",
    "13-2051.00": "13-2099.01",
}


class JobDescriptions:
    """Build the 54-row job-description corpus."""

    def __init__(
        self,
        onet_loader: OnetLoader,
        occupations_config: OccupationsConfig,
    ) -> None:
        self._loader = onet_loader
        self._occupations = occupations_config

    # ------------------------------------------------------------------ public

    def build(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for occ in self._occupations.occupations:
            soc = occ["onet_soc"]
            title = self._title(soc)
            description = self._description(soc)
            skills = self._top_skills(soc, k=_TOP_K_SKILLS)
            knowledge = self._top_knowledge(soc, k=_TOP_K_KNOWLEDGE)
            for phrasing_id, summary, requirements in self._phrasings(
                title=title,
                description=description,
                skills=skills,
                knowledge=knowledge,
            ):
                rows.append(
                    {
                        "occupation_soc": soc,
                        "phrasing_id": phrasing_id,
                        "title": title,
                        "summary": summary,
                        "requirements": requirements,
                    }
                )
        return pd.DataFrame(rows)

    def lint(self, df: pd.DataFrame) -> list[str]:
        violations: list[str] = []
        for idx, row in df.iterrows():
            for field in ("title", "summary", "requirements"):
                text_lower = str(row[field]).lower()
                for word in BANNED_DEMOGRAPHIC_WORDS:
                    pattern = r"\b" + re.escape(word) + r"\b"
                    if re.search(pattern, text_lower):
                        violations.append(
                            f"row={idx} occupation_soc={row['occupation_soc']} "
                            f"phrasing_id={row['phrasing_id']} field={field} "
                            f"token={word!r}"
                        )
        return violations

    def write(self, output_path: Path) -> None:
        df = self.build()
        violations = self.lint(df)
        if violations:
            raise ValueError(
                "Refusing to write job_descriptions.parquet: "
                f"{len(violations)} demographic-signal violation(s) detected:\n  "
                + "\n  ".join(violations)
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

    # ----------------------------------------------------------------- private

    def _title(self, soc: str) -> str:
        df = self._loader.load_occupations()
        row = df.loc[df["onet_soc"] == soc]
        if row.empty:
            raise ValueError(f"SOC not in O*NET: {soc}")
        return str(row.iloc[0]["title"])

    def _description(self, soc: str) -> str:
        df = self._loader.load_occupations()
        row = df.loc[df["onet_soc"] == soc]
        return str(row.iloc[0]["description"])

    def _top_skills(self, soc: str, k: int) -> tuple[str, ...]:
        return self._top_elements(self._loader.load_skills(), self._ratings_soc(soc), k)

    def _top_knowledge(self, soc: str, k: int) -> tuple[str, ...]:
        return self._top_elements(self._loader.load_knowledge(), self._ratings_soc(soc), k)

    @staticmethod
    def _ratings_soc(soc: str) -> str:
        return _PROXY_FOR_RATINGS.get(soc, soc)

    @staticmethod
    def _top_elements(df: pd.DataFrame, soc: str, k: int) -> tuple[str, ...]:
        cell = df.loc[(df["onet_soc"] == soc) & (df["scale_id"] == "IM")].copy()
        if cell.empty:
            return ()
        cell["data_value_num"] = cell["data_value"].astype(float)
        cell = cell.sort_values("data_value_num", ascending=False)
        names: list[str] = cell["element_name"].head(k).tolist()
        return tuple(names)

    @staticmethod
    def _phrasings(
        title: str,
        description: str,
        skills: tuple[str, ...],
        knowledge: tuple[str, ...],
    ) -> list[tuple[int, str, str]]:
        # All literal wording below is hand-authored. No LLM in the loop.

        # Phrasing 0 — canonical (full O*NET description + 6 skills).
        summary_0 = description
        requirements_0 = "Required skills: " + "; ".join(skills) + "."

        # Phrasing 1 — concise hiring-style (first sentence + 3 skills).
        first_sentence = description.split(".", 1)[0].strip() + "."
        summary_1 = "We are hiring for the position of " + title + ". " + first_sentence
        top_three_skills = skills[:3] if len(skills) >= 3 else skills
        requirements_1 = "Must have proven ability in: " + ", ".join(top_three_skills) + "."

        # Phrasing 2 — formal corporate posting (full description + bulleted
        # skills + bulleted knowledge areas).
        summary_2 = "Position: " + title + ".\n\nRole summary:\n" + description
        skills_block = "\n".join("- " + s for s in skills)
        knowledge_block = "\n".join("- " + k for k in knowledge)
        requirements_2 = "Key skills:\n" + skills_block + "\n\nKnowledge areas:\n" + knowledge_block

        return [
            (0, summary_0, requirements_0),
            (1, summary_1, requirements_1),
            (2, summary_2, requirements_2),
        ]
