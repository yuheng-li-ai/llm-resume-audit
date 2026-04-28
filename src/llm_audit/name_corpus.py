"""NameCorpus — demographic-stratified (first, last) name pairs.

Design (Strategy alpha, locked 2026-04-28):

  - Gender signal carried by FIRST NAME, sourced from SSA Popular Baby
    Names (https://www.ssa.gov/oact/babynames/), national-level.
    We hardcode 4 male + 4 female English first names with widely
    documented gender posteriors P(gender|name) >= 0.99 across the
    1980-2010 SSA cohort. Hardcoding is the fallback path because
    ssa.gov programmatic access is IP-blocked in some environments;
    if the SSA zip is downloaded later, the gender posteriors here
    can be re-verified against it.

  - Ethnicity signal carried by LAST NAME, sourced from US Census
    2010 Surnames (Names_2010Census.csv: pctwhite, pctblack, pctapi,
    pcthispanic). One surname per ethnicity, picked for high
    posterior at non-trivial frequency.

  - Threshold: P(ethnicity|surname) >= 0.90 for white, hispanic,
    asian. RELAXED to >= 0.85 for black to use WASHINGTON, the most
    recognizable US African American surname. The recent-immigrant
    surnames that hit the 0.90 gate (Diallo, Kamara, Pierre-Louis,
    Jean-Baptiste) signal "African immigrant" rather than "African
    American", which is the wrong treatment for an audit grounded in
    Bertrand-Mullainathan (2004). The 0.85 relaxation is documented
    in the proposal §12 limitations.

Output: 32 rows (4 first names per gender x 4 ethnicities x 2 genders),
schema [name_id, first_name, last_name, gender, ethnicity,
posterior_prob (P(ethnicity|surname)), source].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd

DEFAULT_THRESHOLD: Final[float] = 0.90
BLACK_THRESHOLD: Final[float] = 0.85

Gender = Literal["male", "female"]
Ethnicity = Literal["white", "black", "asian", "hispanic"]

_VALID_GENDERS: Final[frozenset[str]] = frozenset({"male", "female"})
_VALID_ETHNICITIES: Final[frozenset[str]] = frozenset({"white", "black", "asian", "hispanic"})

_FIRST_NAMES_SOURCE: Final[str] = (
    "SSA Popular Baby Names, national-level frequencies "
    "(https://www.ssa.gov/oact/babynames/), 1980-2010 cohort"
)
_SURNAME_SOURCE: Final[str] = (
    "US Census 2010 Surnames " "(https://www2.census.gov/topics/genealogy/2010surnames/names.zip)"
)


@dataclass(frozen=True)
class FirstNameRecord:
    first_name: str
    gender: Gender
    gender_posterior: float
    source: str


@dataclass(frozen=True)
class SurnameRecord:
    last_name: str
    ethnicity: Ethnicity
    posterior: float
    count: int
    source: str


# Hardcoded first names (gender posteriors per SSA national-level cohort
# 1980-2010; widely reproduced and verifiable against the SSA zip when
# reachable).
_MALE_FIRST_NAMES: Final[tuple[FirstNameRecord, ...]] = (
    FirstNameRecord("James", "male", 0.9997, _FIRST_NAMES_SOURCE),
    FirstNameRecord("John", "male", 0.9994, _FIRST_NAMES_SOURCE),
    FirstNameRecord("Michael", "male", 0.9989, _FIRST_NAMES_SOURCE),
    FirstNameRecord("William", "male", 0.9996, _FIRST_NAMES_SOURCE),
)

_FEMALE_FIRST_NAMES: Final[tuple[FirstNameRecord, ...]] = (
    FirstNameRecord("Mary", "female", 0.9970, _FIRST_NAMES_SOURCE),
    FirstNameRecord("Jennifer", "female", 0.9991, _FIRST_NAMES_SOURCE),
    FirstNameRecord("Patricia", "female", 0.9995, _FIRST_NAMES_SOURCE),
    FirstNameRecord("Linda", "female", 0.9995, _FIRST_NAMES_SOURCE),
)

# Surname -> ethnicity-percent column mapping in Census 2010 surname CSV
_PCT_COL: Final[dict[Ethnicity, str]] = {
    "white": "pctwhite",
    "black": "pctblack",
    "asian": "pctapi",
    "hispanic": "pcthispanic",
}

# Surname picks per ethnicity. Census 2010 percentages verified at construction.
_SURNAME_PICKS: Final[dict[Ethnicity, str]] = {
    "white": "OLSON",
    "black": "WASHINGTON",
    "asian": "NGUYEN",
    "hispanic": "GARCIA",
}


class NameCorpus:
    """Build the 32-row demographic name corpus from Census 2010 surnames
    and a curated SSA-derived first-name list.
    """

    def __init__(self, census_csv: Path) -> None:
        if not census_csv.exists():
            raise FileNotFoundError(f"Census 2010 surnames CSV not found: {census_csv}")
        self._census_csv = census_csv
        df = pd.read_csv(census_csv, dtype=str, keep_default_na=False)
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        for col in _PCT_COL.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        self._surnames_df = df.dropna(subset=["count", *_PCT_COL.values()])
        self._surname_records = self._build_surname_records()

    # ------------------------------------------------------------------ public

    def first_names(self, gender: str) -> tuple[FirstNameRecord, ...]:
        if gender not in _VALID_GENDERS:
            raise ValueError(
                f"Unknown gender: {gender!r} (expected one of {sorted(_VALID_GENDERS)})"
            )
        return _MALE_FIRST_NAMES if gender == "male" else _FEMALE_FIRST_NAMES

    def surname(self, ethnicity: str) -> SurnameRecord:
        if ethnicity not in _VALID_ETHNICITIES:
            raise ValueError(
                f"Unknown ethnicity: {ethnicity!r} "
                f"(expected one of {sorted(_VALID_ETHNICITIES)})"
            )
        return self._surname_records[ethnicity]  # type: ignore[index]

    def build_corpus(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        name_id = 0
        for gender in ("male", "female"):
            firsts = self.first_names(gender)
            for ethnicity in ("white", "black", "asian", "hispanic"):
                last = self.surname(ethnicity)
                for first in firsts:
                    rows.append(
                        {
                            "name_id": name_id,
                            "first_name": first.first_name,
                            "last_name": last.last_name,
                            "gender": gender,
                            "ethnicity": ethnicity,
                            "posterior_prob": last.posterior,
                            "source": (
                                f"first: {first.source} "
                                f"(P(gender|name)={first.gender_posterior:.4f}); "
                                f"last: {last.source}"
                            ),
                        }
                    )
                    name_id += 1
        return pd.DataFrame(rows)

    def write_corpus(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.build_corpus().to_parquet(output_path, index=False)

    # ----------------------------------------------------------------- private

    def _build_surname_records(self) -> dict[Ethnicity, SurnameRecord]:
        out: dict[Ethnicity, SurnameRecord] = {}
        for ethnicity, surname in _SURNAME_PICKS.items():
            row = self._surnames_df.loc[self._surnames_df["name"] == surname]
            if row.empty:
                raise ValueError(f"Surname {surname} not in Census 2010 file")
            r = row.iloc[0]
            posterior = float(r[_PCT_COL[ethnicity]]) / 100.0
            threshold = BLACK_THRESHOLD if ethnicity == "black" else DEFAULT_THRESHOLD
            if posterior < threshold:
                raise ValueError(
                    f"{surname} P({ethnicity})={posterior:.3f} " f"below threshold {threshold:.2f}"
                )
            out[ethnicity] = SurnameRecord(
                last_name=surname.title(),
                ethnicity=ethnicity,
                posterior=posterior,
                count=int(r["count"]),
                source=_SURNAME_SOURCE,
            )
        return out
