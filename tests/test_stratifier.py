"""Tests for Stratifier (Phase 4.2-4.3).

Stratifier produces ~8,000 TreatmentCell objects from the 450 base résumés
× 48 demographic cells × 18 occupations product space, with ~9 reps per
micro-cell + slight s_signal=False oversample.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_audit.stratifier import Stratifier
from llm_audit.utils.io import SeedConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_RESUMES_PARQUET = _REPO_ROOT / "data" / "processed" / "base_resumes.parquet"
_SEEDS_PATH = _REPO_ROOT / "config" / "seeds.toml"


@pytest.fixture(scope="session")
def base_resumes() -> pd.DataFrame:
    if not _BASE_RESUMES_PARQUET.exists():
        pytest.skip(f"base_resumes.parquet not at {_BASE_RESUMES_PARQUET} (run Phase 1.2.4)")
    return pd.read_parquet(_BASE_RESUMES_PARQUET)


@pytest.fixture(scope="session")
def stratifier() -> Stratifier:
    return Stratifier(seed_config=SeedConfig(_SEEDS_PATH))


class TestEnumeration:
    def test_full_enumeration_has_21600_cells(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        # 450 base résumés × 48 demographic cells = 21,600 candidate cells
        full = stratifier.enumerate_full(base_resumes)
        assert len(full) == 450 * 48


class TestStratify:
    def test_returns_target_size_cells(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        assert 7800 <= len(cells) <= 8200

    def test_cell_ids_are_unique_and_sequential(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        ids = [c.cell_id for c in cells]
        assert ids == list(range(len(cells)))

    def test_all_18_occupations_present(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        assert len({c.occupation_soc for c in cells}) == 18

    def test_each_micro_cell_has_about_9_reps(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        # 18 occupations × 48 demographic cells = 864 micro-cells; 8000/864 ≈ 9.26
        cells = stratifier.stratify(base_resumes, n_target=8000)
        df = pd.DataFrame(
            [(c.occupation_soc, c.t_g, c.t_e, c.t_p, c.s_signal) for c in cells],
            columns=["soc", "t_g", "t_e", "t_p", "s"],
        )
        per_micro = df.groupby(["soc", "t_g", "t_e", "t_p", "s"]).size()
        assert per_micro.min() >= 8, f"min reps per micro-cell = {per_micro.min()}"
        assert per_micro.max() <= 11, f"max reps per micro-cell = {per_micro.max()}"

    def test_slight_oversample_on_s_signal_false(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        n_s_true = sum(1 for c in cells if c.s_signal)
        n_s_false = sum(1 for c in cells if not c.s_signal)
        assert (
            n_s_false > n_s_true
        ), f"expected oversample on s=False, got s=True={n_s_true}, s=False={n_s_false}"

    def test_demographic_cells_are_well_formed(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        # 48 demographic cells = 2 t_g × 4 t_e × 3 t_p × 2 s_signal
        demo_cells = {(c.t_g, c.t_e, c.t_p, c.s_signal) for c in cells}
        assert len(demo_cells) == 48

    def test_resume_id_assigned_within_correct_occupation(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        cells = stratifier.stratify(base_resumes, n_target=8000)
        soc_to_resume_ids = base_resumes.groupby("occupation_soc")["resume_id"].apply(set).to_dict()
        for c in cells:
            assert c.resume_id in soc_to_resume_ids[c.occupation_soc], (
                f"cell {c.cell_id}: resume_id {c.resume_id} not in occupation "
                f"{c.occupation_soc}"
            )


class TestDeterminism:
    def test_same_seed_yields_identical_output(
        self, stratifier: Stratifier, base_resumes: pd.DataFrame
    ) -> None:
        a = stratifier.stratify(base_resumes, n_target=500)
        b = stratifier.stratify(base_resumes, n_target=500)
        keys_a = [(c.resume_id, c.t_g, c.t_e, c.t_p, c.s_signal) for c in a]
        keys_b = [(c.resume_id, c.t_g, c.t_e, c.t_p, c.s_signal) for c in b]
        assert keys_a == keys_b
