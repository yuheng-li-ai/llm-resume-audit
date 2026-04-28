"""O*NET 28.1 ingestion.

`OnetLoader` reads the tab-separated text bundle published by
https://www.onetcenter.org/database.html and exposes typed pandas
DataFrames with snake_case column names.

Tables loaded:
    - Occupation Data.txt
    - Task Statements.txt
    - Skills.txt
    - Abilities.txt
    - Knowledge.txt

Each table is loaded lazily on first access and cached on the instance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

_OCCUPATION_FILE: Final[str] = "Occupation Data.txt"
_TASKS_FILE: Final[str] = "Task Statements.txt"
_SKILLS_FILE: Final[str] = "Skills.txt"
_ABILITIES_FILE: Final[str] = "Abilities.txt"
_KNOWLEDGE_FILE: Final[str] = "Knowledge.txt"

_REQUIRED_FILES: Final[tuple[str, ...]] = (
    _OCCUPATION_FILE,
    _TASKS_FILE,
    _SKILLS_FILE,
    _ABILITIES_FILE,
    _KNOWLEDGE_FILE,
)

_OCCUPATION_RENAMES: Final[dict[str, str]] = {
    "O*NET-SOC Code": "onet_soc",
    "Title": "title",
    "Description": "description",
}

_TASKS_RENAMES: Final[dict[str, str]] = {
    "O*NET-SOC Code": "onet_soc",
    "Task ID": "task_id",
    "Task": "task",
    "Task Type": "task_type",
    "Incumbents Responding": "incumbents_responding",
    "Date": "date",
    "Domain Source": "domain_source",
}

_RATING_RENAMES: Final[dict[str, str]] = {
    "O*NET-SOC Code": "onet_soc",
    "Element ID": "element_id",
    "Element Name": "element_name",
    "Scale ID": "scale_id",
    "Data Value": "data_value",
    "N": "n",
    "Standard Error": "standard_error",
    "Lower CI Bound": "lower_ci_bound",
    "Upper CI Bound": "upper_ci_bound",
    "Recommend Suppress": "recommend_suppress",
    "Not Relevant": "not_relevant",
    "Date": "date",
    "Domain Source": "domain_source",
}


class OnetLoader:
    """Read-only loader for the O*NET 28.1 text bundle."""

    def __init__(self, data_dir: Path) -> None:
        if not data_dir.exists():
            raise FileNotFoundError(f"O*NET data directory does not exist: {data_dir}")
        for required in _REQUIRED_FILES:
            if not (data_dir / required).exists():
                raise FileNotFoundError(
                    f"Required O*NET file missing: {required} (looked in {data_dir})"
                )
        self._data_dir = data_dir
        self._cache: dict[str, pd.DataFrame] = {}

    @property
    def data_dir(self) -> Path:
        return self._data_dir

    def load_occupations(self) -> pd.DataFrame:
        return self._load_table(_OCCUPATION_FILE, _OCCUPATION_RENAMES)

    def load_tasks(self) -> pd.DataFrame:
        return self._load_table(_TASKS_FILE, _TASKS_RENAMES)

    def load_skills(self) -> pd.DataFrame:
        return self._load_table(_SKILLS_FILE, _RATING_RENAMES)

    def load_abilities(self) -> pd.DataFrame:
        return self._load_table(_ABILITIES_FILE, _RATING_RENAMES)

    def load_knowledge(self) -> pd.DataFrame:
        return self._load_table(_KNOWLEDGE_FILE, _RATING_RENAMES)

    def load_tasks_for_soc(self, onet_soc: str) -> pd.DataFrame:
        tasks = self.load_tasks()
        return tasks.loc[tasks["onet_soc"] == onet_soc]

    def _load_table(self, filename: str, renames: dict[str, str]) -> pd.DataFrame:
        if filename in self._cache:
            return self._cache[filename]
        path = self._data_dir / filename
        logger.debug("Loading O*NET table: %s", path)
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            encoding="utf-8",
        )
        missing = set(renames) - set(df.columns)
        if missing:
            raise ValueError(f"Expected columns missing in {filename}: {sorted(missing)}")
        df = df.rename(columns=renames)
        self._cache[filename] = df
        return df
