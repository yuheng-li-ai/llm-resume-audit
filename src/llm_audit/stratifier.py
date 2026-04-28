"""Stratifier — enumerate the full résumé × treatment grid and draw an
~8,000-cell stratified subsample per proposal §4.

Layout:
    18 occupations × 48 demographic cells (= 2 t_g × 4 t_e × 3 t_p × 2 s_signal)
    = 864 micro-cells. Each occupation has 25 base résumés in
    base_resumes.parquet, so the full enumeration is 450 résumés × 48
    demographic cells = 21,600 candidate cells.

Sampling rule (deterministic, seed from config/seeds.toml):
    - 9 résumé picks per micro-cell (without replacement from the 25
      base résumés in that occupation) -> 9 × 864 = 7,776 base cells.
    - Add +1 extra rep on a uniform random subset of (s_signal=False)
      micro-cells to reach n_target. With n_target=8000 this is about
      224 extra cells, all on the s=False side -> slight H_3 oversample.
    - Total ~7,992-8,208 cells (within ±2% of 8,000 per proposal §4 AC).

Output: list[TreatmentCell] with sequential cell_id starting at 0.
"""

from __future__ import annotations

import random
from typing import Final, get_args

import pandas as pd

from llm_audit.schema import YearsExpBracket
from llm_audit.treatment_injector import Ethnicity, Gender, TreatmentCell
from llm_audit.utils.io import SeedConfig

_GENDERS: Final[tuple[Gender, ...]] = ("male", "female")
_ETHNICITIES: Final[tuple[Ethnicity, ...]] = ("white", "black", "asian", "hispanic")
_BRACKETS: Final[tuple[YearsExpBracket, ...]] = get_args(YearsExpBracket)
_S_SIGNALS: Final[tuple[bool, ...]] = (True, False)

_REPS_PER_MICRO_CELL: Final[int] = 9


class Stratifier:
    """Build the stratified ~8,000-cell treatment-assignment plan."""

    def __init__(self, seed_config: SeedConfig) -> None:
        self._master_seed = seed_config.get("treatment_injector", "master")
        self._subsample_seed = seed_config.get("treatment_injector", "subsample")

    def enumerate_full(self, base_resumes_df: pd.DataFrame) -> pd.DataFrame:
        """Cartesian product of base résumés × 48 demographic cells."""
        demo_rows = [
            {"t_g": g, "t_e": e, "t_p": p, "s_signal": s}
            for g in _GENDERS
            for e in _ETHNICITIES
            for p in _BRACKETS
            for s in _S_SIGNALS
        ]
        demo = pd.DataFrame(demo_rows)
        return base_resumes_df.merge(demo, how="cross")

    def stratify(
        self,
        base_resumes_df: pd.DataFrame,
        n_target: int = 8000,
    ) -> list[TreatmentCell]:
        rng = random.Random(self._subsample_seed + n_target)
        out: list[TreatmentCell] = []
        cell_id = 0

        soc_to_rows: dict[str, list[pd.Series]] = {}
        for _, row in base_resumes_df.iterrows():
            soc_to_rows.setdefault(row["occupation_soc"], []).append(row)
        socs = sorted(soc_to_rows.keys())

        # Base pass: ~9 résumé picks per (soc × demographic) micro-cell
        for soc in socs:
            pool = soc_to_rows[soc]
            for g in _GENDERS:
                for e in _ETHNICITIES:
                    for p in _BRACKETS:
                        for s in _S_SIGNALS:
                            picks = rng.sample(pool, k=min(_REPS_PER_MICRO_CELL, len(pool)))
                            for row in picks:
                                out.append(
                                    TreatmentCell(
                                        cell_id=cell_id,
                                        resume_id=int(row["resume_id"]),
                                        t_g=g,
                                        t_e=e,
                                        t_p=p,
                                        s_signal=s,
                                        occupation_soc=soc,
                                        education_tier=row["education_tier"],
                                    )
                                )
                                cell_id += 1

        # Oversample s_signal=False until we hit n_target
        deficit = max(0, n_target - len(out))
        if deficit > 0:
            s_false_micro_cells = [
                (soc, g, e, p)
                for soc in socs
                for g in _GENDERS
                for e in _ETHNICITIES
                for p in _BRACKETS
            ]
            rng.shuffle(s_false_micro_cells)
            for i in range(min(deficit, len(s_false_micro_cells))):
                soc, g, e, p = s_false_micro_cells[i]
                pool = soc_to_rows[soc]
                row = rng.choice(pool)
                out.append(
                    TreatmentCell(
                        cell_id=cell_id,
                        resume_id=int(row["resume_id"]),
                        t_g=g,
                        t_e=e,
                        t_p=p,
                        s_signal=False,
                        occupation_soc=soc,
                        education_tier=row["education_tier"],
                    )
                )
                cell_id += 1

        return out
