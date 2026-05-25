"""Phase 7.2 — GRF CATE estimator via econml.grf.CausalForest.

Implements proposal §6.2:

    tau(x) = E[Y(t) - Y(t') | X = x]

Fits one CausalForest per (axis x non-baseline level) contrast vs. the
locked baseline (T_g=male, T_e=white, T_p=early_career). Six contrasts
in total:
    t_g: female vs male
    t_e: black/hispanic/asian vs white
    t_p: mid_career/late_career vs early_career

Covariates: X = [J, M, S, years_exp_bracket, education_tier] per proposal,
explicitly NOT including other demographic axes (avoids conditioning on
other treatments).

CI caveat. econml.grf.CausalForest exposes no cluster argument; the 95%
intervals returned by `predict(..., interval=True)` assume IID rows. The
audit's row unit is (cell x model) and each base resume contributes ~36
rows (one per cell x 2 models), so observations sharing a resume_id are
correlated. Point estimates (tau_hat, mean, percentiles) are unaffected;
CI widths in cate_<axis>_<level>.csv and the p01/p99 bands in
cate_percentiles.pdf are likely anticonservative. Cluster-corrected CIs
require a downstream block-bootstrap pass over resume_id; not implemented
in this phase.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from econml.grf import CausalForest

from llm_audit.analysis.ols import DEFAULT_BASELINES

DEFAULT_COVARIATE_COLS: Final[tuple[str, ...]] = (
    "occupation_soc",
    "model_id",
    "s_signal",
    "years_exp_bracket",
    "education_tier",
)
DEFAULT_CATEGORICAL_COVARIATES: Final[tuple[str, ...]] = (
    "occupation_soc",
    "model_id",
    "years_exp_bracket",
    "education_tier",
)


@dataclass(frozen=True)
class CATEConfig:
    """Column-name + reference-level + forest config."""

    score_col: str = "hiring_score"
    occupation_col: str = "occupation_soc"
    model_col: str = "model_id"
    signal_col: str = "s_signal"
    treatment_cols: tuple[str, ...] = ("t_g", "t_e", "t_p")
    covariate_cols: tuple[str, ...] = DEFAULT_COVARIATE_COLS
    categorical_covariates: tuple[str, ...] = DEFAULT_CATEGORICAL_COVARIATES
    baselines: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BASELINES))
    n_estimators: int = 100
    min_samples_leaf: int = 5
    random_state: int = 7
    n_jobs: int = -1


@dataclass(frozen=True)
class CATEResult:
    """One fitted (axis x level) contrast."""

    axis: str
    contrast_level: str
    baseline: str
    tau_hat: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    feature_importances: pd.Series
    design_frame: pd.DataFrame


class CATEAnalyzer:
    """Fit per-contrast CausalForests and expose tidy result tables."""

    def __init__(self, config: CATEConfig | None = None) -> None:
        self._config = config or CATEConfig()

    @property
    def config(self) -> CATEConfig:
        return self._config

    def fit(
        self,
        scores: pd.DataFrame,
        treatments: pd.DataFrame,
        base_resumes: pd.DataFrame,
    ) -> list[CATEResult]:
        cfg = self._config
        design = self._build_design_frame(scores, treatments, base_resumes)
        results: list[CATEResult] = []
        for axis in cfg.treatment_cols:
            ref = cfg.baselines[axis]
            non_baseline = [v for v in sorted(design[axis].astype(str).unique()) if v != ref]
            for level in non_baseline:
                results.append(self._fit_one_contrast(design, axis, level, ref))
        return results

    def _build_design_frame(
        self,
        scores: pd.DataFrame,
        treatments: pd.DataFrame,
        base_resumes: pd.DataFrame,
    ) -> pd.DataFrame:
        cfg = self._config
        missing_cells = set(scores["cell_id"]).difference(treatments["cell_id"])
        if missing_cells:
            raise ValueError(
                f"{len(missing_cells)} cell_id values in scores have no treatment record "
                f"(first few: {sorted(missing_cells)[:5]})"
            )

        treat_cols = [
            "cell_id",
            "resume_id",
            *cfg.treatment_cols,
            cfg.signal_col,
            cfg.occupation_col,
        ]
        score_cols = ["cell_id", cfg.score_col, cfg.model_col]
        merged = scores[score_cols].merge(
            treatments[treat_cols], on="cell_id", how="inner", validate="m:1"
        )

        missing_resumes = set(merged["resume_id"]).difference(base_resumes["resume_id"])
        if missing_resumes:
            raise ValueError(
                f"{len(missing_resumes)} resume_id values in treatments have no base_resume "
                f"record (first few: {sorted(missing_resumes)[:5]})"
            )

        base_cols = ["resume_id", "years_exp_bracket", "education_tier"]
        merged = merged.merge(base_resumes[base_cols], on="resume_id", how="left", validate="m:1")

        required = [
            cfg.score_col,
            *cfg.treatment_cols,
            cfg.signal_col,
            cfg.occupation_col,
            "years_exp_bracket",
            "education_tier",
        ]
        cleaned = merged.dropna(subset=required).reset_index(drop=True)
        n_dropped = len(merged) - len(cleaned)
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} row(s) with NaN in required columns "
                f"({required}); effective N reduced from {len(merged)} to "
                f"{len(cleaned)}",
                stacklevel=2,
            )
        return cleaned.assign(**{cfg.signal_col: cleaned[cfg.signal_col].astype(bool).astype(int)})

    def _encode_X(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """One-hot encode categoricals (drop_first); numeric columns kept as float."""
        cfg = self._config
        frames: list[pd.DataFrame] = []
        names: list[str] = []
        for col in cfg.covariate_cols:
            if col in cfg.categorical_covariates:
                dummies = pd.get_dummies(
                    df[col].astype(str), prefix=col, drop_first=True, dtype=float
                )
                frames.append(dummies)
                names.extend(dummies.columns.tolist())
            else:
                frames.append(df[[col]].astype(float))
                names.append(col)
        X = pd.concat(frames, axis=1).to_numpy(dtype=float)
        return X, names

    def _fit_one_contrast(
        self,
        design: pd.DataFrame,
        axis: str,
        level: str,
        baseline: str,
    ) -> CATEResult:
        cfg = self._config
        sub = design.loc[design[axis].isin([baseline, level])].reset_index(drop=True)
        T = (sub[axis] == level).astype(int).to_numpy()
        y = sub[cfg.score_col].astype(float).to_numpy()
        X, feature_names = self._encode_X(sub)

        forest = CausalForest(
            n_estimators=cfg.n_estimators,
            min_samples_leaf=cfg.min_samples_leaf,
            honest=True,
            inference=True,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        forest.fit(X, T, y)
        point, lower, upper = forest.predict(X, interval=True, alpha=0.05)
        tau_hat = np.asarray(point).reshape(-1)
        ci_low = np.asarray(lower).reshape(-1)
        ci_high = np.asarray(upper).reshape(-1)

        importances = pd.Series(
            np.asarray(forest.feature_importances_).reshape(-1),
            index=feature_names,
            name="importance",
        )

        keep_cols = [
            "cell_id",
            cfg.model_col,
            cfg.occupation_col,
            cfg.signal_col,
            "years_exp_bracket",
            "education_tier",
            axis,
        ]
        design_frame = sub[keep_cols].copy()

        return CATEResult(
            axis=axis,
            contrast_level=level,
            baseline=baseline,
            tau_hat=tau_hat,
            ci_low=ci_low,
            ci_high=ci_high,
            feature_importances=importances,
            design_frame=design_frame,
        )

    def percentile_table(self, results: list[CATEResult]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for r in results:
            tau = r.tau_hat
            rows.append(
                {
                    "axis": r.axis,
                    "contrast": r.contrast_level,
                    "baseline": r.baseline,
                    "n": int(tau.shape[0]),
                    "p01": float(np.percentile(tau, 1)),
                    "p50": float(np.percentile(tau, 50)),
                    "p99": float(np.percentile(tau, 99)),
                    "mean": float(np.mean(tau)),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["axis", "contrast", "baseline", "n", "p01", "p50", "p99", "mean"],
        )

    def importance_table(self, results: list[CATEResult]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for r in results:
            for feat, val in r.feature_importances.items():
                rows.append(
                    {
                        "axis": r.axis,
                        "contrast": r.contrast_level,
                        "feature": str(feat),
                        "importance": float(val),
                    }
                )
        return pd.DataFrame(rows, columns=["axis", "contrast", "feature", "importance"])

    def aggregate(self, result: CATEResult, by: list[str]) -> pd.DataFrame:
        """Group tau_hat by the given covariate columns; return n + mean_tau + std_tau."""
        df = result.design_frame.assign(tau_hat=result.tau_hat)
        return df.groupby(by, as_index=False).agg(
            n=("tau_hat", "size"),
            mean_tau=("tau_hat", "mean"),
            std_tau=("tau_hat", "std"),
        )
