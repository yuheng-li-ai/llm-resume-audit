"""Phase 7.3 runner — multiple-hypothesis correction over OLS demographic contrasts.

Builds 21 pre-specified contrasts (7 OLS demographic main effects on the full
panel + 7 per-model on glm-5 + 7 per-model on glm-4.5) and applies the
Romano-Wolf / List-Shaikh-Xu (2019) step-down family-wise error correction
via cluster bootstrap over resume_id.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet

Writes:
    outputs/tables/mht_adjusted.csv

Pre-registration deviation note. Proposal §6.3 anticipated "approximately 96
pre-specified pairwise contrasts". The exact derivation was never fully
pinned in the proposal. We restrict the MHT correction to the 21 contrasts
the audit's main-effects design naturally identifies (the 7 OLS demographic
coefficients reported in Phase 7.1, plus the same 7 estimated separately on
each of the 2 models, for the per-vendor robustness contrast in §7). The
per-occupation CATE contrasts (Phase 7.2) already report 1/50/99 percentile
bands and are not folded back into the family-wise correction.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from llm_audit.analysis.mht import MHTConfig, MHTCorrector
from llm_audit.analysis.ols import OLSAnalyzer

REPO = Path(__file__).resolve().parents[1]
N_BOOTSTRAP = 1000
RANDOM_STATE = 7


def _coef_z_stats(
    scores: pd.DataFrame, treatments: pd.DataFrame, model_filter: str | None
) -> dict[str, float]:
    """Fit OLS on the given subset and return {contrast_name -> z_stat} for the
    7 demographic coefficients (matches OLSAnalyzer.demographic_table())."""
    if model_filter is not None:
        scores = scores.loc[scores["model_id"] == model_filter]
    analyzer = OLSAnalyzer()
    analyzer.fit(scores, treatments)
    demo = analyzer.demographic_table()
    label = model_filter or "ALL"
    return {
        f"{row['name']}|{label}": float(row["coef"] / row["se"])
        for _, row in demo.iterrows()
        if row["se"] > 0
    }


def _cluster_resample_rows(
    scores: pd.DataFrame,
    treatments: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster-bootstrap on resume_id with replacement.

    Returns (boot_scores, boot_treatments) where rows are duplicated to keep
    intra-cluster correlation, and cell_id is reassigned to unique integers so
    OLSAnalyzer's validate="m:1" merge stays valid.
    """
    resume_ids = treatments["resume_id"].unique()
    chosen = rng.choice(resume_ids, size=len(resume_ids), replace=True)

    boot_treat_parts: list[pd.DataFrame] = []
    boot_score_parts: list[pd.DataFrame] = []
    next_cell_id = 0
    for rid in chosen:
        sub_treat = treatments.loc[treatments["resume_id"] == rid].copy()
        sub_score = scores.loc[scores["cell_id"].isin(sub_treat["cell_id"])].copy()
        # remap cell_id to fresh integers so the (resume_id, cell_id) pairs stay unique
        old_to_new = {int(old): next_cell_id + i for i, old in enumerate(sub_treat["cell_id"])}
        sub_treat = sub_treat.assign(cell_id=sub_treat["cell_id"].map(old_to_new))
        sub_score = sub_score.assign(cell_id=sub_score["cell_id"].map(old_to_new))
        boot_treat_parts.append(sub_treat)
        boot_score_parts.append(sub_score)
        next_cell_id += len(sub_treat)

    boot_treat = pd.concat(boot_treat_parts, ignore_index=True)
    boot_score = pd.concat(boot_score_parts, ignore_index=True)
    return boot_score, boot_treat


def _bootstrap_z_stats_fn(
    scores: pd.DataFrame,
    treatments: pd.DataFrame,
    observed: dict[str, float],
) -> Callable[[int], dict[str, float]]:
    """Build the bootstrap_stats_fn LSX expects."""

    def _draw(seed: int) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        boot_score, boot_treat = _cluster_resample_rows(scores, treatments, rng)
        out: dict[str, float] = {}
        for subset_label in (None, "glm-5", "glm-4.5"):
            zb = _coef_z_stats(boot_score, boot_treat, subset_label)
            for k, v in zb.items():
                if k not in observed:
                    continue
                # Studentized null: (bootstrap z) - (observed z)
                out[k] = v - observed[k]
        return out

    return _draw


def main() -> int:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    print(
        f"Inputs: scores={len(scores)} rows | treatments={len(treatments)} rows | "
        f"clusters={treatments['resume_id'].nunique()}"
    )

    observed: dict[str, float] = {}
    for subset_label in (None, "glm-5", "glm-4.5"):
        observed.update(_coef_z_stats(scores, treatments, subset_label))
    print(f"Observed contrasts: {len(observed)}")

    boot_fn = _bootstrap_z_stats_fn(scores, treatments, observed)
    cfg = MHTConfig(n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE)

    print(f"Running cluster bootstrap (B={cfg.n_bootstrap}) over resume_id...")
    t0 = time.time()
    results = MHTCorrector(cfg).fit(observed, boot_fn)
    print(f"Done in {time.time() - t0:.1f}s")

    df = (
        pd.DataFrame(
            [
                {
                    "contrast_name": r.contrast_name,
                    "observed_stat": r.observed_stat,
                    "raw_p": r.raw_p,
                    "adjusted_p": r.adjusted_p,
                }
                for r in results
            ]
        )
        .sort_values("adjusted_p")
        .reset_index(drop=True)
    )

    out = REPO / "outputs/tables/mht_adjusted.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print()
    print("Adjusted p-values (sorted by adjusted_p ascending):")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 160):
        print(df.to_string(index=False))
    print()
    print(f"Wrote {out.relative_to(REPO)}")
    n_signif_raw = int((df["raw_p"] < 0.05).sum())
    n_signif_adj = int((df["adjusted_p"] < 0.05).sum())
    print(
        f"raw_p < 0.05: {n_signif_raw}/{len(df)} | " f"adjusted_p < 0.05: {n_signif_adj}/{len(df)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
