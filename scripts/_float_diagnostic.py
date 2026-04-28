"""Float-score diagnostic — 50 cells, anchored prompt, temp=0.

Verifies that GLM 4.5 produces a materially continuous score distribution
under the float schema (0.0-100.0, 1+ decimal place). Same 50-cell sample
as the prior integer diagnostic (random_state=99) so distributions are
directly comparable.

Usage (called by run_glm45_diagnostic.sh):
    python -m scripts._float_diagnostic glm45
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from llm_audit.scoring.batch_runner import BatchRunner
from llm_audit.scoring.zhipu_client import ZhipuClient
from llm_audit.treatment_injector import TreatmentCell
from llm_audit.utils.prompts import ScoringPrompt

REPO = Path(__file__).resolve().parents[1]
N = 50
SAMPLE_SEED = 99


def main(tag: str) -> int:
    load_dotenv(REPO / ".env")

    if tag == "glm45":
        client = ZhipuClient(model_id="glm-4.5")
        provider = "zhipu"
        rpm = 30
    else:
        print(
            f"unknown tag: {tag!r} (expected glm45)",
            file=sys.stderr,
        )
        return 2

    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    jobs = pd.read_parquet(REPO / "data/processed/job_descriptions.parquet")
    base = pd.read_parquet(REPO / "data/processed/base_resumes.parquet").set_index("resume_id")

    sample = treatments.sample(n=N, random_state=SAMPLE_SEED).reset_index(drop=True)
    jd0 = jobs.loc[jobs["phrasing_id"] == 0].set_index("occupation_soc")

    prompts = []
    for _, r in sample.iterrows():
        jd = jd0.loc[r["occupation_soc"]]
        jd_text = f"{jd['title']}\n\n{jd['summary']}\n\n{jd['requirements']}"
        cell = TreatmentCell(
            cell_id=int(r["cell_id"]),
            resume_id=int(r["resume_id"]),
            t_g=r["t_g"],
            t_e=r["t_e"],
            t_p=r["t_p"],
            s_signal=bool(r["s_signal"]),
            occupation_soc=r["occupation_soc"],
            education_tier=base.loc[int(r["resume_id"]), "education_tier"],
        )
        prompts.append((cell, ScoringPrompt(job_description=jd_text, resume_text=r["prompt_text"])))

    runner = BatchRunner(
        clients={client.model_id: client},
        rpm_per_provider={provider: rpm},
        max_retries=2,
    )

    scores_out = REPO / f"data/audit/scores_{tag}_float_diag.parquet"
    cost_log = REPO / f"data/audit/cost_log_{tag}_float_diag.csv"
    for p in (scores_out, cost_log):
        p.unlink(missing_ok=True)

    print(f"=== {tag.upper()} (model_id={client.model_id}, rpm={rpm}) ===")
    print(f"Submitting {N} cells...")
    df = runner.run_with_prompts(prompts, [client.model_id], scores_out, cost_log)

    n_valid = df["hiring_score"].notna().sum()
    n_failed = df["error"].notna().sum()
    print(f"Total {len(df)} | valid {n_valid} ({n_valid/len(df)*100:.1f}%) | failed {n_failed}")

    quota_errors = df.loc[
        df["error"].notna() & df["error"].str.contains("429|quota|rate", na=False, case=False)
    ]
    if len(quota_errors):
        print(f"Rate-limit / quota errors: {len(quota_errors)}")
        print(f"  first: {quota_errors.iloc[0]['error'][:160]}")

    scores = df["hiring_score"].dropna()
    if len(scores):
        unique_vals = sorted(scores.unique().tolist())
        print(f"unique values: {len(unique_vals)}")
        print(f"  min/max  = {scores.min():.2f} / {scores.max():.2f}")
        print(
            f"  mean / median / std = {scores.mean():.2f} / {scores.median():.2f} / {scores.std():.3f}"
        )
        q = [scores.quantile(p) for p in (0.10, 0.25, 0.50, 0.75, 0.90)]
        print("  quartiles 10/25/50/75/90 = " + " / ".join(f"{v:.1f}" for v in q))
        vc = scores.round(1).value_counts().sort_values(ascending=False).head(10)
        print(f"  top-10 score values (rounded to 0.1): {dict(vc)}")
        verdict = (
            "MATERIALLY CONTINUOUS"
            if len(unique_vals) >= 15
            else ("MARGINAL" if len(unique_vals) >= 8 else "COARSE")
        )
        print(f"  VERDICT: {verdict}")
        print("  3 sample outputs:")
        for i in range(min(3, len(df))):
            r = df.iloc[i]
            if pd.notna(r["hiring_score"]):
                rationale = (r["rationale"] or "")[:120].replace("\n", " ")
                print(f"    cell={r['cell_id']} score={r['hiring_score']:.2f} '{rationale}...'")
    else:
        print("ALL FAILED")

    ct = runner.cost_tracker
    print(f"  cost: {ct.total_cost()}  tokens: {ct.total_tokens()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
