"""Phase 7.1 runner — fit OLS ATE on the main batch and write tables.

Reads:
    data/audit/scores.parquet
    data/processed/treatment_assignments.parquet

Writes:
    outputs/tables/ols_ate.csv                 (full coefficient table)
    outputs/tables/ols_ate_demographic.csv     (AC subset)
    outputs/tables/ols_ate.tex                 (LaTeX subset)
    outputs/tables/ols_ate_joint_f.json        (joint demographic F-test)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from llm_audit.analysis.ols import OLSAnalyzer

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    scores = pd.read_parquet(REPO / "data/audit/scores.parquet")
    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")

    analyzer = OLSAnalyzer()
    results = analyzer.fit(scores, treatments)

    out_dir = REPO / "outputs/tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    full = analyzer.coefficient_table()
    demo = analyzer.demographic_table()
    joint = analyzer.joint_f_test()
    latex = analyzer.latex_table(only_demographic=True)

    full.to_csv(out_dir / "ols_ate.csv", index=False)
    demo.to_csv(out_dir / "ols_ate_demographic.csv", index=False)
    (out_dir / "ols_ate.tex").write_text(latex)
    (out_dir / "ols_ate_joint_f.json").write_text(json.dumps(joint, indent=2))

    n_clusters = treatments.loc[
        treatments["cell_id"].isin(scores["cell_id"]), "resume_id"
    ].nunique()
    print(f"N obs: {int(results.nobs)} | clusters: {n_clusters}")
    print(f"R^2: {results.rsquared:.4f} | adj R^2: {results.rsquared_adj:.4f}")
    print(f"Cov type: {results.cov_type}")
    print()
    print("Demographic coefficients (cluster-robust SE):")
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}"):
        print(demo.to_string(index=False))
    print()
    print(
        f"Joint F-test of demographic nullity: F({joint['df_num']:.0f}, "
        f"{joint['df_den']:.0f}) = {joint['f']:.3f}, p = {joint['p']:.3e}"
    )
    print()
    print("Wrote:")
    for f in ("ols_ate.csv", "ols_ate_demographic.csv", "ols_ate.tex", "ols_ate_joint_f.json"):
        print(f"  {(out_dir / f).relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
