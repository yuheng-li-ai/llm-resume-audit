"""Phase 7.1 unit tests — OLSAnalyzer with cluster-robust SE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from llm_audit.analysis.ols import DEFAULT_BASELINES, OLSAnalyzer, OLSConfig


def _make_synthetic_panel(
    n_resumes: int = 60,
    cells_per_resume: int = 8,
    seed: int = 7,
    beta_g_female: float = -2.0,
    beta_e_black: float = -5.0,
    beta_e_hispanic: float = -1.5,
    beta_e_asian: float = 0.5,
    beta_p_mid: float = 1.0,
    beta_p_late: float = -0.5,
    beta_signal: float = 3.0,
    sigma: float = 1.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Build (scores, treatments) with known true coefficients for recovery."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    cell_id = 0
    for resume_id in range(n_resumes):
        # Per-resume offset induces intra-cluster correlation (justifies cluster SE)
        resume_offset = float(rng.normal(0.0, 4.0))
        for _ in range(cells_per_resume):
            t_g = str(rng.choice(["male", "female"]))
            t_e = str(rng.choice(["white", "black", "hispanic", "asian"]))
            t_p = str(rng.choice(["early_career", "mid_career", "late_career"]))
            s_signal = bool(rng.choice([False, True]))
            occupation = str(rng.choice(["jobA", "jobB", "jobC"]))
            rows.append(
                {
                    "cell_id": cell_id,
                    "resume_id": resume_id,
                    "t_g": t_g,
                    "t_e": t_e,
                    "t_p": t_p,
                    "s_signal": s_signal,
                    "occupation_soc": occupation,
                    "_offset": resume_offset,
                }
            )
            cell_id += 1
    treatments = pd.DataFrame(rows).drop(columns=["_offset"])
    offsets = pd.DataFrame(rows)["_offset"].to_numpy()

    score_rows: list[dict[str, object]] = []
    for model_id in ("glm-5", "glm-4.5"):
        model_offset = -1.0 if model_id == "glm-4.5" else 0.0
        for i, row in enumerate(rows):
            y = (
                70.0
                + offsets[i]
                + model_offset
                + (beta_g_female if row["t_g"] == "female" else 0.0)
                + (beta_e_black if row["t_e"] == "black" else 0.0)
                + (beta_e_hispanic if row["t_e"] == "hispanic" else 0.0)
                + (beta_e_asian if row["t_e"] == "asian" else 0.0)
                + (beta_p_mid if row["t_p"] == "mid_career" else 0.0)
                + (beta_p_late if row["t_p"] == "late_career" else 0.0)
                + (beta_signal if bool(row["s_signal"]) else 0.0)
                + float(rng.normal(0.0, sigma))
            )
            score_rows.append(
                {
                    "cell_id": int(row["cell_id"]),
                    "model_id": model_id,
                    "hiring_score": float(y),
                }
            )
    scores = pd.DataFrame(score_rows)

    truth = {
        'C(t_g, Treatment(reference="male"))[T.female]': beta_g_female,
        'C(t_e, Treatment(reference="white"))[T.black]': beta_e_black,
        'C(t_e, Treatment(reference="white"))[T.hispanic]': beta_e_hispanic,
        'C(t_e, Treatment(reference="white"))[T.asian]': beta_e_asian,
        'C(t_p, Treatment(reference="early_career"))[T.mid_career]': beta_p_mid,
        'C(t_p, Treatment(reference="early_career"))[T.late_career]': beta_p_late,
        "s_signal": beta_signal,
    }
    return scores, treatments, truth


class TestOLSAnalyzerFit:
    def test_recovers_known_coefficients_within_tolerance(self) -> None:
        scores, treatments, truth = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        coefs = analyzer.coefficient_table().set_index("name")["coef"]
        for name, true_value in truth.items():
            estimated = float(coefs.loc[name])
            assert (
                abs(estimated - true_value) < 1.2
            ), f"{name}: estimated {estimated:.3f} vs true {true_value:.3f}"

    def test_results_property_raises_before_fit(self) -> None:
        analyzer = OLSAnalyzer()
        with pytest.raises(RuntimeError, match="fit"):
            _ = analyzer.results

    def test_uses_cluster_robust_covariance(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        results = analyzer.fit(scores, treatments)
        assert results.cov_type == "cluster"

    def test_raises_on_missing_treatment_for_score_cell(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        # introduce a score row whose cell_id is not in treatments
        scores = pd.concat(
            [
                scores,
                pd.DataFrame([{"cell_id": 999_999, "model_id": "glm-5", "hiring_score": 50.0}]),
            ],
            ignore_index=True,
        )
        analyzer = OLSAnalyzer()
        with pytest.raises(ValueError, match="cell_id values in scores have no treatment"):
            analyzer.fit(scores, treatments)

    def test_handles_object_dtype_signal_column(self) -> None:
        # Some parquet round-trips can yield object-dtype booleans. The
        # analyzer must coerce them safely instead of silently producing
        # all-1s via .astype(int).
        scores, treatments, truth = _make_synthetic_panel()
        treatments = treatments.assign(s_signal=treatments["s_signal"].astype(object))
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        coefs = analyzer.coefficient_table().set_index("name")["coef"]
        assert abs(float(coefs.loc["s_signal"]) - truth["s_signal"]) < 1.0


class TestCoefficientTable:
    def test_table_has_expected_columns(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        df = analyzer.coefficient_table()
        assert list(df.columns) == ["name", "coef", "se", "t", "p", "ci_low", "ci_high"]
        assert (df["se"] > 0).all()
        assert ((df["p"] >= 0) & (df["p"] <= 1)).all()
        assert (df["ci_low"] <= df["ci_high"]).all()

    def test_demographic_table_returns_ac_required_rows(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        demo = analyzer.demographic_table()
        names = set(demo["name"])
        # AC: beta_g (1), beta_e[black|hispanic|asian] (3), beta_p[mid|late] (2), beta_S (1)
        assert len(names) == 7
        assert any("[T.female]" in n for n in names)
        assert any("[T.black]" in n for n in names)
        assert any("[T.hispanic]" in n for n in names)
        assert any("[T.asian]" in n for n in names)
        assert any("[T.mid_career]" in n for n in names)
        assert any("[T.late_career]" in n for n in names)
        assert "s_signal" in names
        assert not any("[T.male]" in n for n in names)
        assert not any("[T.white]" in n for n in names)
        assert not any("[T.early_career]" in n for n in names)


class TestJointFTest:
    def test_default_demographic_joint_test_returns_signif(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        result = analyzer.joint_f_test()
        assert result["f"] > 0
        assert 0 <= result["p"] <= 1
        assert result["df_num"] == 6  # 1 gender + 3 ethnicity + 2 age
        assert result["n_restrictions"] == 6
        assert result["p"] < 0.001  # synthetic effects are large

    def test_unknown_prefix_raises(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        with pytest.raises(ValueError, match="no coefficients matched"):
            analyzer.joint_f_test(term_prefixes=["C(nonexistent,"])


class TestLatexTable:
    def test_latex_renders_with_demographic_rows_only(self) -> None:
        scores, treatments, _ = _make_synthetic_panel()
        analyzer = OLSAnalyzer()
        analyzer.fit(scores, treatments)
        latex = analyzer.latex_table(only_demographic=True)
        assert "\\begin{tabular}" in latex
        assert "Estimate" in latex
        assert "[T.black]" in latex
        # to_latex(escape=True) converts underscores -> \_ for safe compile
        assert "s\\_signal" in latex
        assert "95\\% CI" in latex


class TestOLSConfig:
    def test_default_baselines_match_proposal(self) -> None:
        cfg = OLSConfig()
        assert cfg.baselines == DEFAULT_BASELINES
        assert cfg.baselines["t_g"] == "male"
        assert cfg.baselines["t_e"] == "white"
        assert cfg.baselines["t_p"] == "early_career"
        assert cfg.cluster_col == "resume_id"
