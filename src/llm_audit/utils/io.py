"""IO helpers — config loading, path resolution.

`SeedConfig` is the single resolver for random seeds; per CHECKLIST rule 7,
every random draw must pull its seed from config/seeds.toml via this class.

`OccupationsConfig` parses the locked 18-occupation panel from
config/occupations.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


class OccupationsConfig:
    """Read-only loader for the 18-occupation panel at config/occupations.toml.

    Schema (per occupation block):
        [[occupation]]
        soc          = "XX-YYYY"
        onet_soc     = "XX-YYYY.NN"
        title        = "..."
        stereotype   = "male" | "female" | "neutral"
        skill_tier   = "high" | "mid" | "low"
        oes_may_2024 = <int>
    """

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Occupations config not found: {path}")
        self._path = path
        with path.open("rb") as f:
            data = tomllib.load(f)
        occs = data.get("occupation", [])
        if not isinstance(occs, list):
            raise ValueError(f"{path}: top-level [[occupation]] must be a list of tables")
        self._occupations: tuple[dict[str, Any], ...] = tuple(occs)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def occupations(self) -> tuple[dict[str, Any], ...]:
        return self._occupations

    def onet_socs(self) -> tuple[str, ...]:
        return tuple(o["onet_soc"] for o in self._occupations)


class SeedConfig:
    """Read-only loader for the seed registry at config/seeds.toml.

    Schema:
        [seeds.<module>]
        <key> = <int>
    """

    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Seed registry not found: {path}")
        self._path = path
        with path.open("rb") as f:
            data = tomllib.load(f)
        self._modules: dict[str, dict[str, Any]] = data.get("seeds", {})

    @property
    def path(self) -> Path:
        return self._path

    def get(self, module: str, key: str) -> int:
        try:
            module_seeds = self._modules[module]
        except KeyError as exc:
            raise KeyError(f"Unknown module in seeds.toml: {module!r}") from exc
        try:
            value = module_seeds[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown seed key {key!r} in module {module!r} "
                f"(available: {sorted(module_seeds)})"
            ) from exc
        if not isinstance(value, int):
            raise TypeError(
                f"Seed at [seeds.{module}].{key} must be int, got {type(value).__name__}"
            )
        return value

    def module_seeds(self, module: str) -> dict[str, int]:
        try:
            module_seeds = self._modules[module]
        except KeyError as exc:
            raise KeyError(f"Unknown module in seeds.toml: {module!r}") from exc
        return {k: int(v) for k, v in module_seeds.items()}
