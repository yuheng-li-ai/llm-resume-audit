"""Phase 8 — robustness checks.

8.1 Permutation placebo (proposal §7-1)
---------------------------------------
Shuffle treatment labels WITHIN each resume cluster and re-fit the OLS
specification. Under the null of no treatment-outcome association, the
placebo coefficients should be centered on zero. A meaningful departure
flags either a data-leak in the merge pipeline or a coding error in the
main analysis. Reports a two-sided permutation p-value computed as the
fraction of placebo coefficients with |coef| >= |observed_coef|.

The shuffle is within-cluster so the cluster structure (resume_id ->
fixed years_exp, education, occupation) is preserved; only the demographic
treatment labels are randomized, breaking the score-treatment link
without disturbing the rest of the design.

8.3 Temporal-split drift check (proposal §7-3, substitute)
----------------------------------------------------------
Split the score parquet into N equal-size time windows by the
`retrieved_at` ISO-8601 timestamp, refit OLS demographic coefficients
in each window, and compare coefficient stability across windows. If a
demographic coefficient shifts materially across windows (more than a
within-window SE), it flags vendor-side model behaviour change during
the collection window. The CHECKLIST originally specified weekly
calibration-résumé re-scores; that design assumed calibration cells
were injected during the main batch (they were not), so the temporal
split serves as the substantive substitute on the actual instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from llm_audit.analysis.ols import OLSAnalyzer

_SHUFFLE_COLS: Final[tuple[str, ...]] = ("t_g", "t_e", "t_p", "s_signal")


@dataclass(frozen=True)
class PermutationPlaceboConfig:
    """Configuration for the within-cluster permutation placebo."""

    n_permutations: int = 500
    cluster_col: str = "resume_id"
    random_state: int = 7


@dataclass(frozen=True)
class PermutationPlaceboResult:
    """One demographic coefficient + its placebo distribution."""

    contrast_name: str
    observed_coef: float
    placebo_coefs: np.ndarray  # shape (n_permutations,)
    placebo_two_sided_p: float


class PermutationPlaceboAnalyzer:
    """Within-cluster permutation placebo for OLS demographic coefficients."""

    def __init__(self, config: PermutationPlaceboConfig | None = None) -> None:
        self._config = config or PermutationPlaceboConfig()

    @property
    def config(self) -> PermutationPlaceboConfig:
        return self._config

    def fit(self, scores: pd.DataFrame, treatments: pd.DataFrame) -> list[PermutationPlaceboResult]:
        cfg = self._config
        observed_analyzer = OLSAnalyzer()
        observed_analyzer.fit(scores, treatments)
        observed_demo = observed_analyzer.demographic_table().set_index("name")["coef"]
        contrast_names = list(observed_demo.index)

        rng = np.random.default_rng(cfg.random_state)
        placebo_matrix = np.zeros((cfg.n_permutations, len(contrast_names)), dtype=float)

        for b in range(cfg.n_permutations):
            permuted = self._permute_within_cluster(treatments, rng)
            placebo_analyzer = OLSAnalyzer()
            placebo_analyzer.fit(scores, permuted)
            placebo_demo = placebo_analyzer.demographic_table().set_index("name")["coef"]
            for j, name in enumerate(contrast_names):
                placebo_matrix[b, j] = float(placebo_demo.loc[name])

        results: list[PermutationPlaceboResult] = []
        for j, name in enumerate(contrast_names):
            obs = float(observed_demo.loc[name])
            placebo = placebo_matrix[:, j]
            # Two-sided permutation p with (1+count)/(B+1) bias correction
            count = int(np.sum(np.abs(placebo) >= abs(obs)))
            p = (1.0 + count) / (cfg.n_permutations + 1.0)
            results.append(
                PermutationPlaceboResult(
                    contrast_name=name,
                    observed_coef=obs,
                    placebo_coefs=placebo,
                    placebo_two_sided_p=float(p),
                )
            )
        return results

    def _permute_within_cluster(
        self, treatments: pd.DataFrame, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Shuffle treatment label vectors within each cluster.

        Within each resume_id, rows keep their cell_id / resume_id /
        occupation_soc, but the (t_g, t_e, t_p, s_signal) tuples are
        permuted across rows of that cluster.
        """
        cfg = self._config
        out = treatments.copy()
        for _, idx in treatments.groupby(cfg.cluster_col).indices.items():
            perm = rng.permutation(len(idx))
            for col in _SHUFFLE_COLS:
                values = treatments[col].values[idx]
                out.iloc[idx, out.columns.get_loc(col)] = values[perm]
        return out

    def summary_table(self, results: list[PermutationPlaceboResult]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for r in results:
            rows.append(
                {
                    "contrast_name": r.contrast_name,
                    "observed_coef": r.observed_coef,
                    "placebo_mean": float(r.placebo_coefs.mean()),
                    "placebo_std": float(r.placebo_coefs.std(ddof=1)),
                    "placebo_two_sided_p": r.placebo_two_sided_p,
                    "n_permutations": int(r.placebo_coefs.shape[0]),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "contrast_name",
                "observed_coef",
                "placebo_mean",
                "placebo_std",
                "placebo_two_sided_p",
                "n_permutations",
            ],
        )


# ----------------------------------------------------------------------
# 8.3 Temporal-split drift check
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalDriftConfig:
    """Configuration for the temporal-split drift check."""

    n_windows: int = 2
    time_col: str = "retrieved_at"


@dataclass(frozen=True)
class TemporalDriftResult:
    """OLS coefficient table for one time window."""

    window_label: str
    window_start: str
    window_end: str
    n_obs: int
    coefficient_table: pd.DataFrame  # demographic_table() output


class TemporalDriftAnalyzer:
    """Refit OLS demographic coefficients per time window and compare."""

    def __init__(self, config: TemporalDriftConfig | None = None) -> None:
        self._config = config or TemporalDriftConfig()

    @property
    def config(self) -> TemporalDriftConfig:
        return self._config

    def fit(self, scores: pd.DataFrame, treatments: pd.DataFrame) -> list[TemporalDriftResult]:
        cfg = self._config
        if cfg.time_col not in scores.columns:
            raise ValueError(
                f"scores must contain `{cfg.time_col}` column; have {list(scores.columns)}"
            )
        if cfg.n_windows < 2:
            raise ValueError(f"n_windows must be >= 2; got {cfg.n_windows}")
        sorted_ts = scores[cfg.time_col].sort_values().to_numpy()
        bins = np.quantile(
            np.arange(len(sorted_ts)),
            np.linspace(0.0, 1.0, cfg.n_windows + 1),
        ).astype(int)
        results: list[TemporalDriftResult] = []
        for w in range(cfg.n_windows):
            lo, hi = bins[w], bins[w + 1]
            cutoff_lo = sorted_ts[lo]
            cutoff_hi = sorted_ts[min(hi, len(sorted_ts) - 1)]
            sub = scores.loc[
                (scores[cfg.time_col] >= cutoff_lo) & (scores[cfg.time_col] <= cutoff_hi)
            ]
            analyzer = OLSAnalyzer()
            analyzer.fit(sub, treatments)
            demo = analyzer.demographic_table()
            results.append(
                TemporalDriftResult(
                    window_label=f"w{w + 1}_of_{cfg.n_windows}",
                    window_start=str(cutoff_lo),
                    window_end=str(cutoff_hi),
                    n_obs=int(len(sub)),
                    coefficient_table=demo,
                )
            )
        return results

    def comparison_table(self, results: list[TemporalDriftResult]) -> pd.DataFrame:
        """Wide table: rows = contrasts, cols = window coefs + max-min spread."""
        frames: list[pd.DataFrame] = []
        for r in results:
            t = r.coefficient_table[["name", "coef", "se"]].copy()
            t.columns = ["name", f"{r.window_label}_coef", f"{r.window_label}_se"]
            frames.append(t.set_index("name"))
        wide = pd.concat(frames, axis=1).reset_index()
        coef_cols = [c for c in wide.columns if c.endswith("_coef")]
        wide["max_minus_min"] = wide[coef_cols].max(axis=1) - wide[coef_cols].min(axis=1)
        return wide
