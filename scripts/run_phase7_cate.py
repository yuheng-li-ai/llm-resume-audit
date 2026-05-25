"""Phase 7.2 runner — fit GRF CATE on the main batch and write artefacts.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet
    data/processed/base_resumes.parquet
    config/occupations.toml          (stereotype + skill_tier mapping)

Writes (per the 6 (axis x non-baseline level) contrasts):
    outputs/tables/cate_<axis>_<level>.csv               per-row tau + CI
    outputs/tables/cate_percentiles.csv                  p01/p50/p99/mean
    outputs/tables/cate_importance.csv                   long-form importances
    outputs/tables/cate_by_occupation.csv                wide: 18 occ x 6 contrasts
    outputs/tables/cate_by_stereotype_tier.csv           wide: 9 cells x 6 contrasts
    outputs/tables/cate_marginal.csv                     marginal mean/SD per contrast
    outputs/figures/cate_<axis>_<level>_importance.pdf   top-15 features bar chart
    outputs/figures/cate_<axis>_<level>_heatmap.pdf      occupation x model heatmap
    outputs/figures/cate_percentiles.pdf                 strip plot
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from llm_audit.analysis.cate import CATEAnalyzer, CATEResult  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOP_K_FEATURES = 15


def _load_occupation_meta(path: Path) -> pd.DataFrame:
    """Build occupation_soc -> (stereotype, skill_tier) lookup.

    Data parquets use the O*NET-suffixed form (e.g. "15-1252.00") while
    the TOML's `soc` field is the short form. Use `onet_soc` for the join key.
    """
    with path.open("rb") as f:
        cfg = tomllib.load(f)
    rows = [
        {
            "occupation_soc": entry["onet_soc"],
            "stereotype": entry["stereotype"],
            "skill_tier": entry["skill_tier"],
        }
        for entry in cfg["occupation"]
    ]
    return pd.DataFrame(rows)


def _write_per_row_csv(result: CATEResult, out_dir: Path) -> Path:
    df = result.design_frame.assign(
        tau_hat=result.tau_hat, ci_low=result.ci_low, ci_high=result.ci_high
    )
    path = out_dir / f"cate_{result.axis}_{result.contrast_level}.csv"
    df.to_csv(path, index=False)
    return path


def _plot_importance(result: CATEResult, out_dir: Path) -> Path:
    top = result.feature_importances.sort_values(ascending=False).head(TOP_K_FEATURES)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(top))))
    ax.barh(top.index[::-1], top.values[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title(
        f"CATE feature importance — {result.axis} = {result.contrast_level} "
        f"(vs {result.baseline})"
    )
    fig.tight_layout()
    path = out_dir / f"cate_{result.axis}_{result.contrast_level}_importance.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_heatmap(result: CATEResult, out_dir: Path) -> Path:
    df = result.design_frame.assign(tau_hat=result.tau_hat)
    pivot = df.pivot_table(
        values="tau_hat", index="occupation_soc", columns="model_id", aggfunc="mean"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(5, max(4, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Mean tau_hat by occupation x model — {result.axis} = {result.contrast_level}")
    fig.colorbar(im, ax=ax, label="mean tau_hat")
    fig.tight_layout()
    path = out_dir / f"cate_{result.axis}_{result.contrast_level}_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_percentiles(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(table))))
    labels = [f"{row.axis}={row.contrast}" for row in table.itertuples()]
    y = range(len(table))
    ax.hlines(y, table["p01"], table["p99"], color="grey", lw=2)
    ax.plot(table["p50"], list(y), "o", color="black", label="median")
    ax.plot(table["p01"], list(y), "|", color="blue", label="p01")
    ax.plot(table["p99"], list(y), "|", color="red", label="p99")
    ax.axvline(0, ls="--", color="black", lw=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("tau_hat (hiring-score points)")
    ax.set_title("CATE 1/50/99 percentiles per contrast")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _aggregate_by_occupation(analyzer: CATEAnalyzer, results: list[CATEResult]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for r in results:
        agg = analyzer.aggregate(r, by=["occupation_soc"])
        agg = agg.assign(axis=r.axis, contrast=r.contrast_level)
        frames.append(agg[["axis", "contrast", "occupation_soc", "n", "mean_tau"]])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(values="mean_tau", index="occupation_soc", columns=["axis", "contrast"])
    return wide.reset_index()


def _aggregate_by_stereotype_tier(
    analyzer: CATEAnalyzer,
    results: list[CATEResult],
    occupation_meta: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for r in results:
        df = r.design_frame.assign(tau_hat=r.tau_hat).merge(
            occupation_meta, on="occupation_soc", how="left"
        )
        agg = (
            df.groupby(["stereotype", "skill_tier"], as_index=False)
            .agg(n=("tau_hat", "size"), mean_tau=("tau_hat", "mean"))
            .assign(axis=r.axis, contrast=r.contrast_level)
        )
        frames.append(agg[["axis", "contrast", "stereotype", "skill_tier", "n", "mean_tau"]])
    long = pd.concat(frames, ignore_index=True)
    wide = long.pivot_table(
        values="mean_tau",
        index=["stereotype", "skill_tier"],
        columns=["axis", "contrast"],
    )
    return wide.reset_index()


def _marginal_table(results: list[CATEResult]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for r in results:
        rows.append(
            {
                "axis": r.axis,
                "contrast": r.contrast_level,
                "n": int(r.tau_hat.shape[0]),
                "mean_tau": float(r.tau_hat.mean()),
                "std_tau": float(r.tau_hat.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    base_resumes = pd.read_parquet(REPO / "data/processed/base_resumes.parquet")
    occupation_meta = _load_occupation_meta(REPO / "config/occupations.toml")

    print(
        f"Inputs: scores={len(scores)} | treatments={len(treatments)} | "
        f"base_resumes={len(base_resumes)} | occupations={len(occupation_meta)}"
    )

    analyzer = CATEAnalyzer()
    print(
        f"Fitting {len(analyzer.config.treatment_cols)} axes "
        f"(n_estimators={analyzer.config.n_estimators}, "
        f"min_samples_leaf={analyzer.config.min_samples_leaf})..."
    )
    results = analyzer.fit(scores, treatments, base_resumes)
    print(f"Fit {len(results)} contrasts.")

    tables_dir = REPO / "outputs/tables"
    figures_dir = REPO / "outputs/figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for r in results:
        written.append(_write_per_row_csv(r, tables_dir))
        written.append(_plot_importance(r, figures_dir))
        written.append(_plot_heatmap(r, figures_dir))

    percentiles = analyzer.percentile_table(results)
    percentiles.to_csv(tables_dir / "cate_percentiles.csv", index=False)
    written.append(tables_dir / "cate_percentiles.csv")

    importance_long = analyzer.importance_table(results)
    importance_long.to_csv(tables_dir / "cate_importance.csv", index=False)
    written.append(tables_dir / "cate_importance.csv")

    by_occ = _aggregate_by_occupation(analyzer, results)
    by_occ.to_csv(tables_dir / "cate_by_occupation.csv", index=False)
    written.append(tables_dir / "cate_by_occupation.csv")

    by_st = _aggregate_by_stereotype_tier(analyzer, results, occupation_meta)
    by_st.to_csv(tables_dir / "cate_by_stereotype_tier.csv", index=False)
    written.append(tables_dir / "cate_by_stereotype_tier.csv")

    marginal = _marginal_table(results)
    marginal.to_csv(tables_dir / "cate_marginal.csv", index=False)
    written.append(tables_dir / "cate_marginal.csv")

    _plot_percentiles(percentiles, figures_dir / "cate_percentiles.pdf")
    written.append(figures_dir / "cate_percentiles.pdf")

    print()
    print("Per-contrast percentiles:")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(percentiles.to_string(index=False))
    print()
    print("Marginal mean / std:")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(marginal.to_string(index=False))
    print()
    print(
        "NOTE: CIs in cate_<axis>_<level>.csv assume IID rows. The audit's row "
        "unit is (cell x model); resume_id correlation is not modeled by GRF, "
        "so reported CI widths are likely anticonservative. Cluster-corrected "
        "intervals require a downstream block-bootstrap pass (not in 7.2)."
    )
    print()
    print(f"Wrote {len(written)} artefacts:")
    for p in written:
        print(f"  {p.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
