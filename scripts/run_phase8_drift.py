"""Phase 8.3 runner — temporal-split drift check (proposal §7-3 substitute).

Split scores.parquet into N equal-size time windows by `retrieved_at`,
refit OLS demographic coefficients per window, compare stability.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet

Writes:
    outputs/tables/drift_temporal_split.csv     wide table (per-window coefs + max-min)
    outputs/figures/drift_temporal.pdf          per-coef trajectory across windows
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from llm_audit.analysis.robustness import (  # noqa: E402
    TemporalDriftAnalyzer,
    TemporalDriftConfig,
    TemporalDriftResult,
)

REPO = Path(__file__).resolve().parents[1]
N_WINDOWS = 3


def _short_name(full_name: str) -> str:
    if "[T." in full_name:
        return full_name.split("[T.")[1].rstrip("]")
    return full_name


def _plot_trajectories(
    comparison: pd.DataFrame, results: list[TemporalDriftResult], out_path: Path
) -> None:
    contrasts = comparison["name"].tolist()
    window_labels = [r.window_label for r in results]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(contrasts))))
    for name in contrasts:
        row = comparison.loc[comparison["name"] == name].iloc[0]
        coefs = [row[f"{w}_coef"] for w in window_labels]
        ses = [row[f"{w}_se"] for w in window_labels]
        x = range(len(window_labels))
        ax.errorbar(
            list(x),
            coefs,
            yerr=[1.96 * s for s in ses],
            marker="o",
            capsize=3,
            label=_short_name(name),
        )
    ax.axhline(0, ls="--", color="black", lw=0.6)
    ax.set_xticks(list(range(len(window_labels))))
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Coefficient (hiring-score points) ± 95% CI")
    ax.set_title(f"Phase 8.3 — Temporal-split drift check ({len(window_labels)} windows)")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    print(
        f"Inputs: scores={len(scores)} | treatments={len(treatments)} | "
        f"window range = [{scores['retrieved_at'].min()}, {scores['retrieved_at'].max()}]"
    )

    cfg = TemporalDriftConfig(n_windows=N_WINDOWS)
    an = TemporalDriftAnalyzer(cfg)
    results = an.fit(scores, treatments)
    print(f"Fit {len(results)} time windows:")
    for r in results:
        print(f"  {r.window_label}: n={r.n_obs} | {r.window_start} -> {r.window_end}")

    comparison = an.comparison_table(results)
    tables_dir = REPO / "outputs/tables"
    figures_dir = REPO / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "drift_temporal_split.csv"
    out_pdf = figures_dir / "drift_temporal.pdf"
    comparison.to_csv(out_csv, index=False)
    _plot_trajectories(comparison, results, out_pdf)

    print()
    print("Comparison (per-window coefficients, max - min spread):")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 200):
        print(comparison.to_string(index=False))
    print()
    print(f"Wrote {out_csv.relative_to(REPO)}")
    print(f"Wrote {out_pdf.relative_to(REPO)}")

    spread = comparison["max_minus_min"]
    flagged = comparison.loc[spread > 1.0]
    if len(flagged):
        print()
        print("WARNING: coefficient spread > 1.0 score points across windows:")
        print(flagged[["name", "max_minus_min"]].to_string(index=False))
    else:
        print()
        print("PASS: all demographic coefficients stable across windows (max-min ≤ 1.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
