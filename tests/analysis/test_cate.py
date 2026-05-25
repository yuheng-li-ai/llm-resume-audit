"""Phase 7.2 unit tests — CATEAnalyzer (GRF CATE via econml.grf.CausalForest)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_audit.analysis.cate import CATEAnalyzer, CATEConfig
from llm_audit.analysis.ols import DEFAULT_BASELINES


def _make_synthetic_panel_with_heterogeneity(
    n_resumes: int = 40,
    cells_per_resume: int = 8,
    seed: int = 7,
    sigma: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    """Build (scores, treatments, base_resumes, truth_by_axis_occ).

    Heterogeneous female effect by occupation:
      jobA: female = -6.0   (large penalty)
      jobB: female = -1.0   (small penalty)
      jobC: female =  0.0   (none)
    Other treatment effects are flat across covariates.
    """
    rng = np.random.default_rng(seed)
    occupations = ["jobA", "jobB", "jobC"]
    years_brackets = ["early_career", "mid_career", "late_career"]
    edu_tiers = ["associate", "bachelor", "master"]

    base_rows: list[dict[str, object]] = []
    for rid in range(n_resumes):
        base_rows.append(
            {
                "resume_id": rid,
                "occupation_soc": str(rng.choice(occupations)),
                "years_exp_bracket": str(rng.choice(years_brackets)),
                "education_tier": str(rng.choice(edu_tiers)),
            }
        )
    base = pd.DataFrame(base_rows)

    treat_rows: list[dict[str, object]] = []
    cell_id = 0
    for rid in range(n_resumes):
        resume_occ = base.loc[rid, "occupation_soc"]
        for _ in range(cells_per_resume):
            t_g = str(rng.choice(["male", "female"]))
            t_e = str(rng.choice(["white", "black", "hispanic", "asian"]))
            t_p = str(rng.choice(["early_career", "mid_career", "late_career"]))
            s_signal = bool(rng.choice([False, True]))
            treat_rows.append(
                {
                    "cell_id": cell_id,
                    "resume_id": rid,
                    "t_g": t_g,
                    "t_e": t_e,
                    "t_p": t_p,
                    "s_signal": s_signal,
                    "occupation_soc": resume_occ,
                }
            )
            cell_id += 1
    treatments = pd.DataFrame(treat_rows)

    def _female_tau(occ: str) -> float:
        return {"jobA": -6.0, "jobB": -1.0, "jobC": 0.0}[occ]

    score_rows: list[dict[str, object]] = []
    for model_id in ("glm-5", "glm-4.5"):
        model_offset = -1.0 if model_id == "glm-4.5" else 0.0
        for r in treat_rows:
            occ = str(r["occupation_soc"])
            y = (
                70.0
                + model_offset
                + (_female_tau(occ) if r["t_g"] == "female" else 0.0)
                + (-3.0 if r["t_e"] == "black" else 0.0)
                + (2.0 if r["t_p"] == "late_career" else 0.0)
                + (1.0 if bool(r["s_signal"]) else 0.0)
                + float(rng.normal(0.0, sigma))
            )
            score_rows.append(
                {
                    "cell_id": int(r["cell_id"]),
                    "model_id": model_id,
                    "hiring_score": float(y),
                }
            )
    scores = pd.DataFrame(score_rows)

    truth = {
        "t_g_female_by_occ": {"jobA": -6.0, "jobB": -1.0, "jobC": 0.0},
    }
    return scores, treatments, base, truth


class TestCATEConfig:
    def test_defaults_match_proposal(self) -> None:
        cfg = CATEConfig()
        assert cfg.baselines == DEFAULT_BASELINES
        assert cfg.baselines["t_g"] == "male"
        assert cfg.baselines["t_e"] == "white"
        assert cfg.baselines["t_p"] == "early_career"
        for col in (
            "occupation_soc",
            "model_id",
            "s_signal",
            "years_exp_bracket",
            "education_tier",
        ):
            assert col in cfg.covariate_cols
        assert cfg.n_estimators >= 50
        assert cfg.min_samples_leaf >= 1
        assert cfg.random_state is not None


class TestBuildDesignFrame:
    def test_design_frame_has_all_columns(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        an = CATEAnalyzer()
        df = an._build_design_frame(scores, treat, base)
        for col in (
            "cell_id",
            "model_id",
            "hiring_score",
            "t_g",
            "t_e",
            "t_p",
            "s_signal",
            "occupation_soc",
            "years_exp_bracket",
            "education_tier",
            "resume_id",
        ):
            assert col in df.columns, col
        assert len(df) == len(scores)

    def test_raises_on_missing_treatment_for_score_cell(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        scores = pd.concat(
            [
                scores,
                pd.DataFrame([{"cell_id": 999_999, "model_id": "glm-5", "hiring_score": 50.0}]),
            ],
            ignore_index=True,
        )
        an = CATEAnalyzer()
        with pytest.raises(ValueError, match="cell_id values in scores have no treatment"):
            an._build_design_frame(scores, treat, base)

    def test_raises_on_missing_base_resume(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        base = base.iloc[1:].reset_index(drop=True)  # drop resume_id=0
        an = CATEAnalyzer()
        with pytest.raises(ValueError, match="resume_id"):
            an._build_design_frame(scores, treat, base)


class TestEncodeX:
    def test_encode_returns_numeric_no_nan(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        an = CATEAnalyzer()
        df = an._build_design_frame(scores, treat, base)
        X, names = an._encode_X(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(names)
        assert not np.isnan(X).any()
        assert np.issubdtype(X.dtype, np.floating)

    def test_encoded_columns_have_variance(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        an = CATEAnalyzer()
        df = an._build_design_frame(scores, treat, base)
        X, _ = an._encode_X(df)
        assert X.std(axis=0).min() > 0


class TestCATEAnalyzerFit:
    def test_returns_one_result_per_non_baseline_level(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        assert len(results) == 1 + 3 + 2
        levels = {(r.axis, r.contrast_level) for r in results}
        assert levels == {
            ("t_g", "female"),
            ("t_e", "black"),
            ("t_e", "hispanic"),
            ("t_e", "asian"),
            ("t_p", "mid_career"),
            ("t_p", "late_career"),
        }

    def test_results_have_aligned_shapes(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        for r in results:
            n = r.tau_hat.shape[0]
            assert r.ci_low.shape == (n,)
            assert r.ci_high.shape == (n,)
            assert len(r.design_frame) == n
            assert (r.ci_low <= r.tau_hat + 1e-9).all()
            assert (r.tau_hat <= r.ci_high + 1e-9).all()

    @pytest.mark.slow
    def test_recovers_known_heterogeneity_within_tolerance(self) -> None:
        scores, treat, base, truth = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=100, min_samples_leaf=5, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        female = next(r for r in results if r.axis == "t_g" and r.contrast_level == "female")
        df = female.design_frame.copy()
        df["tau_hat"] = female.tau_hat
        mean_by_occ = df.groupby("occupation_soc")["tau_hat"].mean()
        a = float(mean_by_occ.loc["jobA"])
        b = float(mean_by_occ.loc["jobB"])
        c = float(mean_by_occ.loc["jobC"])
        assert a < b < c, f"ordering broken: jobA={a:.2f} jobB={b:.2f} jobC={c:.2f}"
        true_a = truth["t_g_female_by_occ"]["jobA"]
        true_b = truth["t_g_female_by_occ"]["jobB"]
        true_c = truth["t_g_female_by_occ"]["jobC"]
        assert abs(a - true_a) < 2.5, f"jobA: {a:.2f} vs true {true_a}"
        assert abs(b - true_b) < 2.5, f"jobB: {b:.2f} vs true {true_b}"
        assert abs(c - true_c) < 2.5, f"jobC: {c:.2f} vs true {true_c}"


class TestFeatureImportance:
    def test_importances_sum_to_one_per_contrast(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        for r in results:
            assert isinstance(r.feature_importances, pd.Series)
            total = float(r.feature_importances.sum())
            assert abs(total - 1.0) < 1e-3, f"{r.axis}/{r.contrast_level}: sum={total:.4f}"


class TestPercentileTable:
    def test_table_columns_and_ordering(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        tbl = an.percentile_table(results)
        assert list(tbl.columns) == [
            "axis",
            "contrast",
            "baseline",
            "n",
            "p01",
            "p50",
            "p99",
            "mean",
        ]
        assert len(tbl) == len(results)
        assert (tbl["p01"] <= tbl["p50"]).all()
        assert (tbl["p50"] <= tbl["p99"]).all()


class TestImportanceTableLong:
    def test_long_form_row_count_and_sum(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        tbl = an.importance_table(results)
        assert set(tbl.columns) == {"axis", "contrast", "feature", "importance"}
        per_contrast = tbl.groupby(["axis", "contrast"])["importance"].sum()
        assert ((per_contrast - 1.0).abs() < 1e-3).all()


class TestAggregate:
    def test_aggregate_by_occupation(self) -> None:
        scores, treat, base, _ = _make_synthetic_panel_with_heterogeneity()
        cfg = CATEConfig(n_estimators=32, min_samples_leaf=10, random_state=7)
        an = CATEAnalyzer(cfg)
        results = an.fit(scores, treat, base)
        female = next(r for r in results if r.axis == "t_g" and r.contrast_level == "female")
        agg = an.aggregate(female, by=["occupation_soc"])
        assert set(agg.columns) >= {"occupation_soc", "n", "mean_tau"}
        assert len(agg) == 3
        assert (agg["n"] > 0).all()
