"""Tests for SeedConfig — TOML-backed seed registry.

Per CHECKLIST rule 7: every random draw uses a seed pulled from
config/seeds.toml; SeedConfig is the single resolver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_audit.utils.io import SeedConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEEDS_PATH = _REPO_ROOT / "config" / "seeds.toml"


@pytest.fixture(scope="session")
def seed_config() -> SeedConfig:
    return SeedConfig(_SEEDS_PATH)


class TestConstruction:
    def test_loads_existing_file(self, seed_config: SeedConfig) -> None:
        assert seed_config.path == _SEEDS_PATH

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            SeedConfig(tmp_path / "nope.toml")


class TestSeedLookup:
    def test_returns_int_for_known_module_key(self, seed_config: SeedConfig) -> None:
        seed = seed_config.get("resume_factory", "master")
        assert isinstance(seed, int)
        assert seed > 0

    def test_raises_on_unknown_module(self, seed_config: SeedConfig) -> None:
        with pytest.raises(KeyError):
            seed_config.get("nonexistent_module", "master")

    def test_raises_on_unknown_key(self, seed_config: SeedConfig) -> None:
        with pytest.raises(KeyError):
            seed_config.get("resume_factory", "nonexistent_key")

    def test_seeds_are_distinct_across_modules(self, seed_config: SeedConfig) -> None:
        a = seed_config.get("resume_factory", "master")
        b = seed_config.get("name_corpus", "master")
        c = seed_config.get("treatment_injector", "master")
        d = seed_config.get("batch_runner", "master")
        assert len({a, b, c, d}) == 4

    def test_module_seeds_returns_full_dict(self, seed_config: SeedConfig) -> None:
        sub = seed_config.module_seeds("resume_factory")
        assert "master" in sub
        assert "faker" in sub
        assert "task_sample" in sub
        assert all(isinstance(v, int) for v in sub.values())
