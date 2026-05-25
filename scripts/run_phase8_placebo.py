"""Phase 8.1 runner — permutation placebo (proposal §7-1).

Shuffles treatment labels within each resume_id cluster, refits OLS, and
checks that the placebo coefficient distribution is centered on zero.
Any meaningful departure flags either a data-leak in the merge pipeline
or a coding error in the main analysis.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet

Writes:
    outputs/tables/placebo_summary.csv          observed coef + placebo mean/SD/p
    outputs/tables/placebo_distribution.csv     long form: contrast_name, permutation, coef
    outputs/figures/placebo.pdf                 histogram per contrast with observed marked
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from llm_audit.analysis.robustness import (  # noqa: E402
    PermutationPlaceboAnalyzer,
    PermutationPlaceboConfig,
    PermutationPlaceboResult,
)

REPO = Path(__file__).resolve().parents[1]
N_PERMUTATIONS = 500
RANDOM_STATE = 7


def _plot_histograms(results: list[PermutationPlaceboResult], out_path: Path) -> None:
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes_flat = axes.flatten() if rows * cols > 1 else [axes]
    for i, r in enumerate(results):
        ax = axes_flat[i]
        ax.hist(r.placebo_coefs, bins=30, color="grey", edgecolor="white")
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.axvline(
            r.observed_coef,
            color="red",
            lw=1.5,
            label=f"observed {r.observed_coef:.3f}",
        )
        ax.set_title(
            r.contrast_name.replace("Treatment(reference=", "ref=").replace("\\", ""),
            fontsize=8,
        )
        ax.set_xlabel("placebo coef")
        ax.legend(loc="best", fontsize=7)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Phase 8.1 — Permutation placebo (within-cluster shuffle)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    print(
        f"Inputs: scores={len(scores)} rows | treatments={len(treatments)} rows | "
        f"clusters={treatments['resume_id'].nunique()}"
    )

    cfg = PermutationPlaceboConfig(n_permutations=N_PERMUTATIONS, random_state=RANDOM_STATE)
    an = PermutationPlaceboAnalyzer(cfg)
    print(f"Running within-cluster permutation placebo (B={cfg.n_permutations})...")
    t0 = time.time()
    results = an.fit(scores, treatments)
    print(f"Done in {time.time() - t0:.1f}s")

    tables_dir = REPO / "outputs/tables"
    figures_dir = REPO / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = an.summary_table(results)
    summary.to_csv(tables_dir / "placebo_summary.csv", index=False)

    long_rows: list[dict[str, object]] = []
    for r in results:
        for b, c in enumerate(r.placebo_coefs.tolist()):
            long_rows.append({"contrast_name": r.contrast_name, "permutation": b, "coef": float(c)})
    pd.DataFrame(long_rows).to_csv(tables_dir / "placebo_distribution.csv", index=False)

    _plot_histograms(results, figures_dir / "placebo.pdf")

    print()
    print("Placebo summary:")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}", "display.width", 160):
        print(summary.to_string(index=False))
    print()
    print(f"Wrote {(tables_dir / 'placebo_summary.csv').relative_to(REPO)}")
    print(f"Wrote {(tables_dir / 'placebo_distribution.csv').relative_to(REPO)}")
    print(f"Wrote {(figures_dir / 'placebo.pdf').relative_to(REPO)}")
    flagged = summary.loc[summary["placebo_mean"].abs() > 0.5]
    if len(flagged) > 0:
        print()
        print("WARNING: placebo_mean exceeded 0.5 for:")
        print(flagged.to_string(index=False))
    else:
        print()
        print("PASS: all placebo means within +/- 0.5 (no leak signature).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
