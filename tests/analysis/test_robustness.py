"""Phase 8 unit tests — robustness checks.

8.1 Permutation placebo (proposal §7-1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_audit.analysis.robustness import (
    PermutationPlaceboAnalyzer,
    PermutationPlaceboConfig,
    PermutationPlaceboResult,
    TemporalDriftAnalyzer,
    TemporalDriftConfig,
    TemporalDriftResult,
)


def _make_panel_with_effect(
    beta_female: float,
    n_resumes: int = 50,
    cells_per_resume: int = 8,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    treat_rows: list[dict[str, object]] = []
    cell_id = 0
    for rid in range(n_resumes):
        for _ in range(cells_per_resume):
            treat_rows.append(
                {
                    "cell_id": cell_id,
                    "resume_id": rid,
                    "t_g": str(rng.choice(["male", "female"])),
                    "t_e": str(rng.choice(["white", "black", "hispanic", "asian"])),
                    "t_p": str(rng.choice(["early_career", "mid_career", "late_career"])),
                    "s_signal": bool(rng.choice([False, True])),
                    "occupation_soc": str(rng.choice(["jobA", "jobB", "jobC"])),
                }
            )
            cell_id += 1
    treatments = pd.DataFrame(treat_rows)

    score_rows: list[dict[str, object]] = []
    for model_id in ("glm-5", "glm-4.5"):
        for r in treat_rows:
            y = 70.0 + (beta_female if r["t_g"] == "female" else 0.0) + float(rng.normal(0.0, 1.0))
            score_rows.append(
                {
                    "cell_id": int(r["cell_id"]),
                    "model_id": model_id,
                    "hiring_score": float(y),
                }
            )
    return pd.DataFrame(score_rows), treatments


class TestPermutationPlaceboConfig:
    def test_defaults(self) -> None:
        cfg = PermutationPlaceboConfig()
        assert cfg.n_permutations >= 100
        assert cfg.random_state is not None
        assert cfg.cluster_col == "resume_id"


class TestPermutationPlaceboAPI:
    def test_returns_one_result_per_demographic_coef(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=0.0)
        cfg = PermutationPlaceboConfig(n_permutations=50, random_state=7)
        an = PermutationPlaceboAnalyzer(cfg)
        results = an.fit(scores, treat)
        assert len(results) == 7  # 1 female + 3 ethnicity + 2 age + s_signal
        for r in results:
            assert isinstance(r, PermutationPlaceboResult)
            assert r.placebo_coefs.shape == (cfg.n_permutations,)
            assert 0.0 <= r.placebo_two_sided_p <= 1.0


class TestPlaceboCenteredAtZero:
    def test_null_data_gives_placebo_centered_near_zero(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=0.0, seed=7)
        cfg = PermutationPlaceboConfig(n_permutations=200, random_state=7)
        an = PermutationPlaceboAnalyzer(cfg)
        results = an.fit(scores, treat)
        for r in results:
            placebo_std = float(r.placebo_coefs.std(ddof=1))
            placebo_mean = float(r.placebo_coefs.mean())
            # Center should sit within 1 SD of zero on small-N synthetic data;
            # on real data (N=16K, B=500) the centering is tight (<0.01 SD).
            assert abs(placebo_mean) < placebo_std, (
                f"{r.contrast_name}: placebo_mean={placebo_mean:.3f} "
                f"exceeds 1 SD of placebo dist ({placebo_std:.3f})"
            )

    def test_strong_true_effect_appears_in_placebo_tail(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=-3.0, seed=7)
        cfg = PermutationPlaceboConfig(n_permutations=200, random_state=7)
        an = PermutationPlaceboAnalyzer(cfg)
        results = an.fit(scores, treat)
        female = next(r for r in results if "female" in r.contrast_name)
        assert (
            female.placebo_two_sided_p < 0.05
        ), f"placebo p={female.placebo_two_sided_p:.4f} for true effect of -3.0"

    def test_within_cluster_shuffle_preserves_cluster_count(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=0.0, seed=7)
        cfg = PermutationPlaceboConfig(n_permutations=10, random_state=7)
        an = PermutationPlaceboAnalyzer(cfg)
        rng = np.random.default_rng(0)
        shuffled = an._permute_within_cluster(treat, rng)
        assert len(shuffled) == len(treat)
        assert (shuffled["resume_id"].values == treat["resume_id"].values).all()
        # At least one treatment column gets shuffled
        assert (shuffled["t_g"].values != treat["t_g"].values).any()


class TestSummaryTable:
    def test_summary_columns_and_shape(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=0.0, seed=7)
        cfg = PermutationPlaceboConfig(n_permutations=50, random_state=7)
        an = PermutationPlaceboAnalyzer(cfg)
        results = an.fit(scores, treat)
        df = an.summary_table(results)
        assert list(df.columns) == [
            "contrast_name",
            "observed_coef",
            "placebo_mean",
            "placebo_std",
            "placebo_two_sided_p",
            "n_permutations",
        ]
        assert len(df) == len(results)
        assert (df["n_permutations"] == cfg.n_permutations).all()


# ----------------------------------------------------------------------
# 8.3 Temporal-split drift check
# ----------------------------------------------------------------------


def _make_panel_with_timestamps(
    beta_female: float = -2.0,
    n_resumes: int = 50,
    cells_per_resume: int = 8,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same as _make_panel_with_effect but adds a retrieved_at ISO column."""
    scores, treat = _make_panel_with_effect(
        beta_female=beta_female,
        n_resumes=n_resumes,
        cells_per_resume=cells_per_resume,
        seed=seed,
    )
    # Assign timestamps monotonically across rows so window-splitting is deterministic.
    base = pd.Timestamp("2026-05-01T00:00:00Z")
    scores = scores.copy()
    scores["retrieved_at"] = [
        (base + pd.Timedelta(seconds=i * 30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(len(scores))
    ]
    return scores, treat


class TestTemporalDriftAnalyzerAPI:
    def test_returns_one_result_per_window(self) -> None:
        scores, treat = _make_panel_with_timestamps()
        cfg = TemporalDriftConfig(n_windows=3)
        an = TemporalDriftAnalyzer(cfg)
        results = an.fit(scores, treat)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TemporalDriftResult)
            assert r.n_obs > 0
            assert "coef" in r.coefficient_table.columns

    def test_raises_on_missing_time_col(self) -> None:
        scores, treat = _make_panel_with_effect(beta_female=0.0)
        an = TemporalDriftAnalyzer()
        with pytest.raises(ValueError, match="retrieved_at"):
            an.fit(scores, treat)

    def test_raises_on_invalid_n_windows(self) -> None:
        with pytest.raises(ValueError, match="n_windows must be >= 2"):
            TemporalDriftAnalyzer(TemporalDriftConfig(n_windows=1)).fit(
                *_make_panel_with_timestamps()
            )


class TestComparisonTable:
    def test_comparison_has_per_window_coef_columns(self) -> None:
        scores, treat = _make_panel_with_timestamps()
        cfg = TemporalDriftConfig(n_windows=2)
        an = TemporalDriftAnalyzer(cfg)
        results = an.fit(scores, treat)
        cmp = an.comparison_table(results)
        assert "name" in cmp.columns
        assert "w1_of_2_coef" in cmp.columns
        assert "w2_of_2_coef" in cmp.columns
        assert "max_minus_min" in cmp.columns
        assert (cmp["max_minus_min"] >= 0).all()


class TestStabilityOnStationaryData:
    def test_stable_coefs_have_small_max_minus_min(self) -> None:
        # Same DGP across whole window -> per-window coefs should be similar
        scores, treat = _make_panel_with_timestamps(beta_female=-2.0)
        cfg = TemporalDriftConfig(n_windows=2)
        an = TemporalDriftAnalyzer(cfg)
        results = an.fit(scores, treat)
        cmp = an.comparison_table(results)
        female_row = cmp.loc[cmp["name"].str.contains("female", regex=False)].iloc[0]
        # Two estimates of the same beta_female should agree within ~2 SD
        assert (
            female_row["max_minus_min"] < 2.0
        ), f"female max-min spread {female_row['max_minus_min']:.3f} too large for stationary DGP"
