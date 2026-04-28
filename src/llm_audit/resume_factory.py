"""ResumeFactory — deterministic synthetic résumé construction.

Produces frozen `Resume` objects anchored on O*NET 28.1 task taxonomies
and Faker-generated institutions/employers. No LLM-written prose;
the body string is purely Jinja2 templating over structured fields.

Determinism: every random draw flows from a per-resume RNG seeded by
combining the master seed (config/seeds.toml [seeds.resume_factory]) with
the integer `resume_id`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Final, cast

import jinja2
import pandas as pd
from faker import Faker

from llm_audit.onet_loader import OnetLoader
from llm_audit.schema import (
    EMAIL_PLACEHOLDER,
    PHONE_PLACEHOLDER,
    Certification,
    Contact,
    Education,
    EducationTier,
    Experience,
    ObjectiveSignals,
    Resume,
    ResumeMeta,
    YearsExpBracket,
)
from llm_audit.utils.io import SeedConfig

_TEMPLATE_NAME: Final[str] = "resume.j2"
_REFERENCE_END_YEAR: Final[int] = 2024

_YEARS_BY_BRACKET: Final[dict[str, int]] = {
    "early_career": 5,
    "mid_career": 15,
    "late_career": 25,
}

_DEGREE_BY_TIER: Final[dict[str, str]] = {
    "high_school": "High School Diploma",
    "associate": "A.A.",
    "bachelor": "B.S.",
    "master": "M.S.",
    "doctorate": "Ph.D.",
}

_FIELD_HINT_BY_SOC: Final[dict[str, str]] = {
    "15-1252.00": "Computer Science",
    "11-2022.00": "Business Administration",
    "47-2031.00": "Carpentry / Construction Trades",
    "49-3023.00": "Automotive Technology",
    "53-3032.00": "Commercial Driving",
    "33-9032.00": "Criminal Justice",
    "29-1141.00": "Nursing",
    "23-1011.00": "Law",
    "25-2021.00": "Elementary Education",
    "43-6014.00": "Office Administration",
    "31-1131.00": "Health Sciences",
    "39-9011.00": "Early Childhood Education",
    "13-2011.00": "Accounting",
    "13-2051.00": "Finance",
    "43-3031.00": "Accounting / Bookkeeping",
    "43-4051.00": "Communications",
    "41-2011.00": "General Studies",
    "37-2011.00": "General Studies",
}

_DEFAULT_FIELD: Final[str] = "General Studies"


class ResumeFactory:
    """Build deterministic synthetic résumés from O*NET task taxonomies."""

    def __init__(
        self,
        onet_loader: OnetLoader,
        seed_config: SeedConfig,
        template_dir: Path,
    ) -> None:
        self._loader = onet_loader
        self._seed_config = seed_config
        self._master_seed = seed_config.get("resume_factory", "master")
        self._faker_seed = seed_config.get("resume_factory", "faker")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=False,
            trim_blocks=False,
            lstrip_blocks=False,
            keep_trailing_newline=True,
        )
        self._template = env.get_template(_TEMPLATE_NAME)

    # ------------------------------------------------------------------ public

    def build_one(
        self,
        resume_id: int,
        soc: str,
        years_exp_bracket: YearsExpBracket,
        education_tier: EducationTier,
        with_signals: bool = False,
    ) -> Resume:
        tasks_df = self._loader.load_tasks_for_soc(soc)
        if tasks_df.empty:
            raise ValueError(f"No O*NET tasks for {soc}")

        rng = random.Random(self._master_seed + resume_id)
        faker = Faker(["en_US"])
        faker.seed_instance(self._faker_seed + resume_id)

        years_total = _YEARS_BY_BRACKET[years_exp_bracket]
        end_year = _REFERENCE_END_YEAR
        career_start = end_year - years_total

        n_jobs = rng.choice([2, 3])
        n_bullets = rng.choice([4, 5])
        task_pool = tasks_df["task"].tolist()
        rng.shuffle(task_pool)
        bullets_per_job = self._chunk(task_pool, n_jobs, n_bullets)

        experiences = self._build_experiences(
            soc=soc,
            faker=faker,
            rng=rng,
            career_start=career_start,
            end_year=end_year,
            bullets_per_job=bullets_per_job,
        )

        education = (
            Education(
                institution=f"{faker.last_name()} State University",
                degree=_DEGREE_BY_TIER[education_tier],
                field=_FIELD_HINT_BY_SOC.get(soc, _DEFAULT_FIELD),
                year=career_start - rng.randint(0, 2),
            ),
        )

        skills = self._top_skills_for_soc(soc, k=8)
        summary = self._summary_for_soc(soc)

        signals: ObjectiveSignals | None = None
        certifications: tuple[Certification, ...] = ()
        if with_signals:
            signals = ObjectiveSignals(
                gpa=round(rng.uniform(3.0, 4.0), 2),
                test_score_percentile=round(rng.uniform(80.0, 99.0), 0),
                certification_ids=tuple(
                    f"CERT-{rng.randint(10_000, 99_999)}" for _ in range(rng.randint(0, 2))
                ),
            )

        return Resume(
            meta=ResumeMeta(
                resume_id=resume_id,
                occupation_soc=soc,
                years_exp_bracket=years_exp_bracket,
                education_tier=education_tier,
            ),
            contact=Contact(email=EMAIL_PLACEHOLDER, phone=PHONE_PLACEHOLDER),
            summary=summary,
            education=education,
            experience=experiences,
            skills=skills,
            certifications=certifications,
            objective_signals=signals,
        )

    def render(self, resume: Resume) -> str:
        rendered: str = self._template.render(resume=resume)
        return rendered

    def panel_to_dataframe(self, panel: Sequence[Resume]) -> pd.DataFrame:
        rows = [
            {
                "resume_id": r.meta.resume_id,
                "occupation_soc": r.meta.occupation_soc,
                "years_exp_bracket": r.meta.years_exp_bracket,
                "education_tier": r.meta.education_tier,
                "body_text": self.render(r),
            }
            for r in panel
        ]
        return pd.DataFrame(rows)

    def write_panel(self, panel: Sequence[Resume], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.panel_to_dataframe(panel).to_parquet(output_path, index=False)

    def build_panel(
        self,
        occupations: Sequence[str],
        n_per_occupation: int = 25,
    ) -> tuple[Resume, ...]:
        out: list[Resume] = []
        rid = 0
        for soc in occupations:
            for _ in range(n_per_occupation):
                bracket = self._bracket_for_position(rid)
                tier = self._tier_for_position(rid)
                with_signals = rid % 2 == 0
                out.append(
                    self.build_one(
                        resume_id=rid,
                        soc=soc,
                        years_exp_bracket=bracket,
                        education_tier=tier,
                        with_signals=with_signals,
                    )
                )
                rid += 1
        return tuple(out)

    # ----------------------------------------------------------------- private

    def _build_experiences(
        self,
        soc: str,
        faker: Faker,
        rng: random.Random,
        career_start: int,
        end_year: int,
        bullets_per_job: list[list[str]],
    ) -> tuple[Experience, ...]:
        n_jobs = len(bullets_per_job)
        years_per_job = max(1, (end_year - career_start) // n_jobs)
        cur_end = end_year
        experiences: list[Experience] = []
        for i, bullets in enumerate(bullets_per_job):
            cur_start = max(career_start, cur_end - years_per_job)
            if i == n_jobs - 1:
                cur_start = career_start
            experiences.append(
                Experience(
                    title=self._title_for_soc(soc, seniority_index=i, total=n_jobs),
                    employer=faker.company(),
                    start_year=cur_start,
                    end_year=cur_end,
                    bullets=tuple(bullets),
                )
            )
            cur_end = cur_start
        return tuple(experiences)

    def _summary_for_soc(self, soc: str) -> str:
        df = self._loader.load_occupations()
        rows = df.loc[df["onet_soc"] == soc]
        if rows.empty:
            return ""
        desc = cast(str, rows.iloc[0]["description"])
        # Truncate to first sentence + a bit, capped at ~250 chars
        if len(desc) <= 250:
            return desc
        cut = desc.rfind(".", 0, 250)
        return desc[: cut + 1] if cut > 0 else desc[:250].rstrip() + "..."

    def _top_skills_for_soc(self, soc: str, k: int) -> tuple[str, ...]:
        skills_df = self._loader.load_skills()
        cell = skills_df.loc[
            (skills_df["onet_soc"] == soc) & (skills_df["scale_id"] == "IM")
        ].copy()
        if cell.empty:
            return ()
        cell["data_value_num"] = cell["data_value"].astype(float)
        cell = cell.sort_values("data_value_num", ascending=False)
        names: list[str] = cell["element_name"].head(k).tolist()
        return tuple(names)

    def _title_for_soc(self, soc: str, seniority_index: int, total: int) -> str:
        df = self._loader.load_occupations()
        rows = df.loc[df["onet_soc"] == soc]
        base = cast(str, rows.iloc[0]["title"]) if len(rows) else soc
        if total <= 1 or seniority_index >= total - 1:
            return base
        if seniority_index == 0:
            return f"Senior {base.rstrip('s')}".rstrip()
        return base

    @staticmethod
    def _chunk(pool: list[str], n_chunks: int, chunk_size: int) -> list[list[str]]:
        """Slice the pool into n_chunks lists of up to chunk_size items each."""
        chunks: list[list[str]] = []
        idx = 0
        for _ in range(n_chunks):
            end = min(idx + chunk_size, len(pool))
            chunks.append(pool[idx:end])
            idx = end
            if idx >= len(pool):
                break
        while len(chunks) < n_chunks:
            chunks.append([pool[0]] if pool else [])
        for i, c in enumerate(chunks):
            if not c and pool:
                chunks[i] = [pool[0]]
        return chunks

    @staticmethod
    def _bracket_for_position(rid: int) -> YearsExpBracket:
        brackets: tuple[YearsExpBracket, YearsExpBracket, YearsExpBracket] = (
            "early_career",
            "mid_career",
            "late_career",
        )
        return brackets[rid % 3]

    @staticmethod
    def _tier_for_position(rid: int) -> EducationTier:
        tiers: tuple[EducationTier, EducationTier, EducationTier] = (
            "bachelor",
            "master",
            "associate",
        )
        return tiers[(rid // 3) % 3]
