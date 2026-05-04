"""Phase 7.1 — OLS ATE estimator with cluster-robust SE.

Implements proposal §6.1:

    Y_{ijm} = alpha
              + beta_g * T^g_i
              + beta_e^T * T^e_i
              + beta_p^T * T^p_i
              + beta_S * S_i
              + delta_j        (occupation fixed effect)
              + mu_m           (model fixed effect)
              + epsilon_{ijm}

Estimated by OLS with cluster-robust SE clustered on resume_id (so the
N independent draws are at the base-resume level, not the cell level).
Per proposal locked baselines:
    T_g  reference = male
    T_e  reference = white
    T_p  reference = early_career
    S    coded as bool; the coefficient is the effect of the
         objective-signal-PRESENT condition (S=True) vs ABSENT (S=False).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper

DEFAULT_BASELINES: Final[dict[str, str]] = {
    "t_g": "male",
    "t_e": "white",
    "t_p": "early_career",
}


@dataclass(frozen=True)
class OLSConfig:
    """Column-name + reference-level configuration for the OLS fit."""

    score_col: str = "hiring_score"
    cluster_col: str = "resume_id"
    occupation_col: str = "occupation_soc"
    model_col: str = "model_id"
    treatment_cols: tuple[str, ...] = ("t_g", "t_e", "t_p")
    signal_col: str = "s_signal"
    baselines: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BASELINES))


class OLSAnalyzer:
    """Fit the proposal §6.1 OLS specification and expose result tables."""

    def __init__(self, config: OLSConfig | None = None) -> None:
        self._config = config or OLSConfig()
        self._results: RegressionResultsWrapper | None = None

    @property
    def results(self) -> RegressionResultsWrapper:
        if self._results is None:
            raise RuntimeError("OLSAnalyzer.fit() must be called before results")
        return self._results

    def fit(
        self,
        scores: pd.DataFrame,
        treatments: pd.DataFrame,
    ) -> RegressionResultsWrapper:
        """Merge scores + treatments on cell_id, fit the OLS, return results."""
        cfg = self._config
        df = self._build_design_frame(scores, treatments)
        formula = self._build_formula()
        model = smf.ols(formula=formula, data=df)
        self._results = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df[cfg.cluster_col].to_numpy()},
        )
        return self._results

    def _build_design_frame(
        self,
        scores: pd.DataFrame,
        treatments: pd.DataFrame,
    ) -> pd.DataFrame:
        cfg = self._config
        score_cols = ["cell_id", cfg.score_col, cfg.model_col]
        treatment_cols = [
            "cell_id",
            cfg.cluster_col,
            *cfg.treatment_cols,
            cfg.signal_col,
            cfg.occupation_col,
        ]
        # Pre-merge sanity: any cell_id in scores absent from treatments would
        # be silently dropped by the inner join. In a pre-registered audit
        # the effective N must match the design — refuse rather than drop.
        missing = set(scores["cell_id"]).difference(treatments["cell_id"])
        if missing:
            raise ValueError(
                f"{len(missing)} cell_id values in scores have no treatment record "
                f"(first few: {sorted(missing)[:5]})"
            )
        merged = scores[score_cols].merge(
            treatments[treatment_cols], on="cell_id", how="inner", validate="m:1"
        )
        required = [cfg.score_col, *cfg.treatment_cols, cfg.signal_col, cfg.occupation_col]
        cleaned = merged.dropna(subset=required).reset_index(drop=True)
        # Immutable transform: avoid mutating the merged frame in place.
        return cleaned.assign(**{cfg.signal_col: cleaned[cfg.signal_col].astype(bool).astype(int)})

    def _build_formula(self) -> str:
        cfg = self._config
        treatment_terms = [
            f'C({col}, Treatment(reference="{cfg.baselines[col]}"))' for col in cfg.treatment_cols
        ]
        rhs = " + ".join(
            [
                *treatment_terms,
                cfg.signal_col,
                f"C({cfg.occupation_col})",
                f"C({cfg.model_col})",
            ]
        )
        return f"{cfg.score_col} ~ {rhs}"

    def coefficient_table(self) -> pd.DataFrame:
        """Return tidy DataFrame: name | coef | se | t | p | ci_low | ci_high."""
        res = self.results
        ci = res.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "name": res.params.index,
                "coef": res.params.values,
                "se": res.bse.values,
                "t": res.tvalues.values,
                "p": res.pvalues.values,
                "ci_low": ci[0].values,
                "ci_high": ci[1].values,
            }
        ).reset_index(drop=True)

    def demographic_table(self) -> pd.DataFrame:
        """Filter coefficient_table to AC-required rows.

        Rows: beta_g, beta_e[black|hispanic|asian],
        beta_p[mid_career|late_career], beta_S.
        """
        cfg = self._config
        rows = self.coefficient_table()
        wanted: list[str] = []
        for col in cfg.treatment_cols:
            ref = cfg.baselines[col]
            wanted.extend(
                n
                for n in rows["name"]
                if n.startswith(f"C({col},") and "[T." in n and not n.endswith(f"[T.{ref}]")
            )
        wanted.append(cfg.signal_col)
        return rows[rows["name"].isin(wanted)].reset_index(drop=True)

    def joint_f_test(self, term_prefixes: list[str] | None = None) -> dict[str, float]:
        """F-test of joint nullity of the demographic coefficients.

        term_prefixes: list of formula-term substrings to include
        (default: all demographic treatment columns). Returns dict with
        f, p, df_num, df_den, n_restrictions.
        """
        cfg = self._config
        prefixes = term_prefixes or [f"C({c}," for c in cfg.treatment_cols]
        names = [
            n
            for n in self.results.params.index
            if any(n.startswith(p) for p in prefixes) and "[T." in n
        ]
        if not names:
            raise ValueError(
                f"no coefficients matched prefixes {prefixes}; "
                f"have {list(self.results.params.index)}"
            )
        hypotheses = ", ".join(f"{n} = 0" for n in names)
        f = self.results.f_test(hypotheses)
        return {
            "f": float(f.fvalue),
            "p": float(f.pvalue),
            "df_num": float(f.df_num),
            "df_den": float(f.df_denom),
            "n_restrictions": float(len(names)),
        }

    def latex_table(self, only_demographic: bool = True) -> str:
        """Render the coefficient table as LaTeX.

        `escape=True` is used so coefficient names with characters that
        are LaTeX-special (e.g. underscores in occupation_soc / model_id
        labels, percent signs in number formatting) compile cleanly. The
        column headers are pre-escape plain strings; pandas escapes them
        on render.
        """
        df = self.demographic_table() if only_demographic else self.coefficient_table()
        out = pd.DataFrame(
            {
                "Term": df["name"].astype(str),
                "Estimate": df["coef"].map(lambda v: f"{v:.3f}"),
                "SE": df["se"].map(lambda v: f"({v:.3f})"),
                "p": df["p"].map(lambda v: f"{v:.3g}"),
                "95% CI": [
                    f"[{lo:.2f}, {hi:.2f}]"
                    for lo, hi in zip(df["ci_low"], df["ci_high"], strict=True)
                ],
            }
        )
        return str(out.to_latex(index=False, escape=True))
