"""Phase 5.5 main batch — score 8000 cells x 2 models on Zhipu paid API.

Two-model panel (post-v0.5b-panel-trim, locked):
    - glm-5     (GLM 5.1, paid)
    - glm-4.5   (GLM 4.5, paid)

Single-vendor (Zhipu) so rate-limiting is one shared bucket. RPM=30 per
provider gives ~16000 calls / 30 RPM = ~9 hours of wall-clock time.
No `google.generativeai` or `groq` imports anywhere — `zhipuai` SDK
connects directly to api.bigmodel.cn from mainland China, no VPN needed.

Resumable: BatchRunner skips any (cell_id, model_id) pair already present
in scores.parquet on each chunk call, so killing and restarting the
process re-uses prior work without duplication.

Outputs:
    data/audit/scores.parquet
    data/audit/cost_log.csv

Usage:
    python -m scripts._main_batch [--chunk-size 200] [--limit N]

    --chunk-size N   how many cells per persistence checkpoint (default 200)
    --limit N        run only the first N cells (smoke-test; default = all)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from llm_audit.scoring.batch_runner import BatchRunner
from llm_audit.scoring.zhipu_client import ZhipuClient
from llm_audit.treatment_injector import TreatmentCell
from llm_audit.utils.prompts import ScoringPrompt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RPM = 30
DEFAULT_MODEL_IDS = ["glm-5", "glm-4.5"]
DEFAULT_CHUNK_SIZE = 200


def _build_prompts(
    treatments: pd.DataFrame,
    jobs: pd.DataFrame,
    base: pd.DataFrame,
) -> list[tuple[TreatmentCell, ScoringPrompt]]:
    jd0 = jobs.loc[jobs["phrasing_id"] == 0].set_index("occupation_soc")
    out: list[tuple[TreatmentCell, ScoringPrompt]] = []
    for _, r in treatments.iterrows():
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
        out.append((cell, ScoringPrompt(job_description=jd_text, resume_text=r["prompt_text"])))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cells")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=200.0,
        help=(
            "Hard budget cap for THIS process (RMB/CNY). Checked between chunks; "
            "abort gracefully when exceeded. Prior runs' spend is NOT counted; "
            "the cap is per-process. Default 200.0."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Single model_id to score (e.g. glm-5 or glm-4.5). Output goes to "
            "data/audit/scores_{model}.parquet so two parallel processes (one "
            "per model) don't race on the same file. If omitted, score both "
            "models in one process to data/audit/scores.parquet (back-compat)."
        ),
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_RPM,
        help=(
            "Requests per minute for THIS process. Set lower for glm-5 "
            "(e.g. 5) when running per-model parallel processes to dodge "
            "the account 1302 rate limit. Default 30."
        ),
    )
    args = parser.parse_args()

    load_dotenv(REPO / ".env")

    treatments = pd.read_parquet(REPO / "data/processed/treatment_assignments.parquet")
    jobs = pd.read_parquet(REPO / "data/processed/job_descriptions.parquet")
    base = pd.read_parquet(REPO / "data/processed/base_resumes.parquet").set_index("resume_id")

    if args.limit > 0:
        treatments = treatments.head(args.limit).reset_index(drop=True)

    prompts = _build_prompts(treatments, jobs, base)
    n_cells = len(prompts)

    if args.model is not None:
        model_ids = [args.model]
        suffix = f"_{args.model}"
    else:
        model_ids = list(DEFAULT_MODEL_IDS)
        suffix = ""
    n_calls = n_cells * len(model_ids)
    print(
        f"Cells: {n_cells} | models: {model_ids} | calls: {n_calls} "
        f"| chunk: {args.chunk_size} | RPM: {args.rpm}"
    )

    clients = {mid: ZhipuClient(model_id=mid) for mid in model_ids}
    runner = BatchRunner(
        clients=clients,
        rpm_per_provider={"zhipu": args.rpm},
        max_retries=4,
    )

    scores_out = REPO / f"data/audit/scores{suffix}.parquet"
    cost_log = REPO / f"data/audit/cost_log{suffix}.csv"
    scores_out.parent.mkdir(parents=True, exist_ok=True)

    pending = n_calls
    if scores_out.exists():
        prior = pd.read_parquet(scores_out)
        prior_ok = prior[prior["hiring_score"].notna()]
        prior_keys = {(int(r.cell_id), str(r.model_id)) for r in prior_ok.itertuples()}
        sample_keys = {(p[0].cell_id, mid) for p in prompts for mid in model_ids}
        already_done = len(prior_keys & sample_keys)
        pending = n_calls - already_done
        print(
            f"Resume: {len(prior)} prior rows ({len(prior_ok)} valid) | "
            f"{already_done} skipped | {pending} pending"
        )
    else:
        print(f"Fresh run: {scores_out.name} does not yet exist | {pending} pending")

    chunks = [prompts[i : i + args.chunk_size] for i in range(0, n_cells, args.chunk_size)]
    started = time.monotonic()
    print(f"Budget cap: RMB {args.max_cost:.2f} (this process only)")
    aborted = False
    pbar = tqdm(total=pending, desc="api calls", unit="call", smoothing=0.05)
    for i, chunk in enumerate(chunks):
        runner.run_with_prompts(chunk, model_ids, scores_out, cost_log, progress_cb=pbar.update)
        elapsed = time.monotonic() - started
        done = (i + 1) / len(chunks)
        eta_min = (elapsed / done - elapsed) / 60 if done > 0 else 0
        ct = runner.cost_tracker
        cost_cny = float(ct.total_cost().get("CNY", 0.0))
        pbar.set_postfix_str(
            f"chunk {i+1}/{len(chunks)} | RMB {cost_cny:.2f} | eta {eta_min:.1f}m",
            refresh=True,
        )
        tqdm.write(
            f"  chunk {i+1}/{len(chunks)} | elapsed {elapsed/60:.1f}m "
            f"| eta {eta_min:.1f}m | cost {ct.total_cost()} | tokens {ct.total_tokens()}"
        )
        if cost_cny >= args.max_cost:
            tqdm.write(
                f"\nBUDGET ABORT: spend RMB {cost_cny:.2f} >= cap RMB {args.max_cost:.2f} "
                f"after chunk {i+1}/{len(chunks)}. Persisted rows are intact in "
                f"{scores_out.name}; re-run to resume with a higher --max-cost."
            )
            aborted = True
            break
    pbar.close()

    df = pd.read_parquet(scores_out)
    valid = df["hiring_score"].notna().sum()
    failed = df["error"].notna().sum()
    print(f"\nFinal: {len(df)} rows | valid {valid} ({valid/len(df)*100:.1f}%) | failed {failed}")
    if valid:
        per_model = df.groupby("model_id")["hiring_score"].agg(["count", "mean", "std", "nunique"])
        print("\nPer-model summary:")
        print(per_model.to_string())

    ct = runner.cost_tracker
    print(f"\nTotal cost: {ct.total_cost()}  total tokens: {ct.total_tokens()}")
    if aborted:
        return 2
    return 0 if valid >= 0.99 * len(df) else 1


if __name__ == "__main__":
    sys.exit(main())
