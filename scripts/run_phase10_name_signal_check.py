"""Phase 10.1 runner -- name-signal validity check through Zhipu direct API.

This small manipulation check asks each deployed GLM model to classify the
intended U.S. name-origin cue represented by the exact names used in the audit.
It addresses the validity concern that U.S. surname signals may not be perceived
as intended by a Chinese LLM.

Reads:
    data/processed/name_corpus.parquet
    .env (ZHIPUAI_API_KEY)

Writes:
    outputs/tables/name_signal_validation_full.csv
    outputs/tables/name_signal_validation_full_summary.csv
    outputs/tables/name_signal_validation_full_confusion.csv

Network:
    Zhipu/BigModel only. The httpx client sets trust_env=False, so HTTP_PROXY,
    HTTPS_PROXY, ALL_PROXY, and VPN-local proxy variables are ignored. OpenAI
    environment variables and common proxy variables are also removed from the
    process before the API client is built.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv
from zhipuai import ZhipuAI

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("glm-5", "glm-4.5")
ETHNICITY_OPTIONS = ("white", "black", "hispanic", "asian", "unclear")
LABEL_TO_TREATMENT = {
    "white/european american": "white",
    "african american": "black",
    "hispanic/spanish-latin american": "hispanic",
    "asian/vietnamese": "asian",
    "unclear": "unclear",
}
SURNAME_TO_TREATMENT = {
    "Olson": "white",
    "Washington": "black",
    "Garcia": "hispanic",
    "Nguyen": "asian",
}
PER_REQUEST_TIMEOUT = 30.0
RPM = 30

SYSTEM_PROMPT = (
    "You are applying a resume-audit codebook to name cues. "
    "Given one full name, answer this codebook item: Which intended U.S. "
    "name-cue origin does this full name most likely represent? "
    "Reply with exactly one option and no explanation: "
    "white/European American; African American; "
    "Hispanic/Spanish-Latin American; Asian/Vietnamese; unclear."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
        help="Repeated classifications per name-model pair. Default: 3.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Zhipu model IDs to query. Default: glm-5 glm-4.5.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=RPM,
        help="Maximum requests per minute. Default: 30.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first two names on the first model and write smoke-only files.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Ignore any existing output file and start a new full/checkpoint run.",
    )
    return parser.parse_args()


def _normalise_response(text: str) -> str:
    lower = text.strip().lower()
    if not lower:
        return "unclear"
    normalised = re.sub(r"\s+", " ", lower.replace(";", " ").replace(",", " "))
    for label, treatment in LABEL_TO_TREATMENT.items():
        if label in normalised:
            return treatment
    for option in ETHNICITY_OPTIONS:
        if re.search(rf"\b{re.escape(option)}\b", lower):
            return option
    return "unparseable"


class _RateLimiter:
    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / max(rpm, 1)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last = time.monotonic()


def _load_existing(out_csv: Path, force_clean: bool) -> pd.DataFrame:
    if out_csv.exists() and not force_clean:
        existing = pd.read_csv(out_csv)
        if "inferred_ethnicity" in existing.columns:
            existing = existing.loc[existing["inferred_ethnicity"] != "unparseable"].copy()
        if "raw_response" in existing.columns:
            existing = existing.loc[
                existing["raw_response"].fillna("").astype(str).str.len() > 0
            ].copy()
        return existing
    return pd.DataFrame(
        columns=[
            "name_id",
            "full_name",
            "intended_gender",
            "intended_ethnicity",
            "posterior_prob",
            "model_id",
            "replicate",
            "inferred_ethnicity",
            "raw_response",
            "correct",
        ]
    )


def _write_outputs(
    df: pd.DataFrame,
    out_csv: Path,
    summary_csv: Path,
    confusion_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["model_id", "name_id", "replicate"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    if df.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            df.groupby(["model_id", "intended_ethnicity"], as_index=False)
            .agg(
                n=("correct", "size"),
                accuracy=("correct", "mean"),
                modal_inferred=("inferred_ethnicity", lambda s: s.mode().iat[0]),
                unclear_rate=("inferred_ethnicity", lambda s: (s == "unclear").mean()),
                unparseable_rate=("inferred_ethnicity", lambda s: (s == "unparseable").mean()),
            )
            .sort_values(["model_id", "intended_ethnicity"])
        )
        overall = (
            df.groupby(["model_id"], as_index=False)
            .agg(
                n=("correct", "size"),
                accuracy=("correct", "mean"),
                modal_inferred=("inferred_ethnicity", lambda s: s.mode().iat[0]),
                unclear_rate=("inferred_ethnicity", lambda s: (s == "unclear").mean()),
                unparseable_rate=("inferred_ethnicity", lambda s: (s == "unparseable").mean()),
            )
            .assign(intended_ethnicity="ALL")
        )
        summary = pd.concat([summary, overall], ignore_index=True)
    summary.to_csv(summary_csv, index=False)

    if df.empty:
        confusion = pd.DataFrame()
    else:
        confusion = (
            df.groupby(["model_id", "intended_ethnicity", "inferred_ethnicity"], as_index=False)
            .size()
            .rename(columns={"size": "n"})
            .sort_values(["model_id", "intended_ethnicity", "inferred_ethnicity"])
        )
    confusion.to_csv(confusion_csv, index=False)


def _clear_disallowed_env() -> None:
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ]:
        os.environ.pop(key, None)


def _query_name(client: ZhipuAI, model_id: str, full_name: str) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Full name: {full_name}"},
        ],
        temperature=0.0,
        max_tokens=512,
        timeout=PER_REQUEST_TIMEOUT,
    )
    return str(response.choices[0].message.content or "")


def main() -> int:
    args = _parse_args()
    load_dotenv(REPO / ".env")
    _clear_disallowed_env()
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        print("ZHIPUAI_API_KEY missing from .env or environment", file=sys.stderr)
        return 1
    if args.replicates < 1:
        print("--replicates must be positive", file=sys.stderr)
        return 2

    names = pd.read_parquet(REPO / "data/processed/name_corpus.parquet")
    names = names.assign(full_name=names["first_name"] + " " + names["last_name"])
    surname_map = names["last_name"].map(SURNAME_TO_TREATMENT)
    if surname_map.isna().any():
        missing = sorted(names.loc[surname_map.isna(), "last_name"].unique())
        print(f"Name corpus has surnames outside validation map: {missing}", file=sys.stderr)
        return 2
    if not (surname_map == names["ethnicity"]).all():
        bad = names.loc[surname_map != names["ethnicity"], ["full_name", "ethnicity"]]
        print(f"Name corpus surname mapping mismatch:\n{bad}", file=sys.stderr)
        return 2
    if args.smoke:
        names = names.head(2)
        args.models = [args.models[0]]
        args.replicates = 1
        prefix = "name_signal_validation_smoke"
    else:
        prefix = "name_signal_validation_full"

    out_csv = REPO / f"outputs/tables/{prefix}.csv"
    summary_csv = REPO / f"outputs/tables/{prefix}_summary.csv"
    confusion_csv = REPO / f"outputs/tables/{prefix}_confusion.csv"
    if args.force_clean:
        for path in (out_csv, summary_csv, confusion_csv):
            path.unlink(missing_ok=True)
    prior = _load_existing(out_csv, force_clean=args.force_clean)
    done = {
        (int(r.name_id), str(r.model_id), int(r.replicate))
        for r in prior.itertuples()
        if pd.notna(r.name_id)
    }

    http_client = httpx.Client(timeout=PER_REQUEST_TIMEOUT, trust_env=False)
    client = ZhipuAI(api_key=api_key, timeout=PER_REQUEST_TIMEOUT, http_client=http_client)
    limiter = _RateLimiter(args.rpm)

    rows: list[dict[str, object]] = []
    total = len(names) * len(args.models) * args.replicates
    completed_before = len(done)
    call_i = 0
    started = time.monotonic()
    print(
        f"Name-signal check: {len(names)} names x {len(args.models)} models x "
        f"{args.replicates} replicates = {total} planned rows."
    )
    if completed_before:
        print(f"Resume: {completed_before} prior rows already present.")

    for model_id in args.models:
        for name in names.itertuples(index=False):
            for replicate in range(args.replicates):
                key = (int(name.name_id), str(model_id), int(replicate))
                if key in done:
                    continue
                call_i += 1
                limiter.wait()
                raw = _query_name(client, str(model_id), str(name.full_name))
                inferred = _normalise_response(raw)
                correct = inferred == str(name.ethnicity)
                rows.append(
                    {
                        "name_id": int(name.name_id),
                        "full_name": str(name.full_name),
                        "intended_gender": str(name.gender),
                        "intended_ethnicity": str(name.ethnicity),
                        "posterior_prob": float(name.posterior_prob),
                        "model_id": str(model_id),
                        "replicate": int(replicate),
                        "inferred_ethnicity": inferred,
                        "raw_response": raw.strip(),
                        "correct": bool(correct),
                    }
                )
                done.add(key)
                if call_i % 10 == 0 or args.smoke:
                    frames = [df for df in [prior, pd.DataFrame(rows)] if not df.empty]
                    current = pd.concat(frames, ignore_index=True) if frames else prior.copy()
                    _write_outputs(current, out_csv, summary_csv, confusion_csv)
                    elapsed = time.monotonic() - started
                    print(
                        f"  new calls {call_i} | total rows {len(current)} | "
                        f"elapsed {elapsed / 60:.1f}m"
                    )

    frames = [df for df in [prior, pd.DataFrame(rows)] if not df.empty]
    final = pd.concat(frames, ignore_index=True) if frames else prior.copy()
    final = final.drop_duplicates(["name_id", "model_id", "replicate"], keep="last")
    _write_outputs(final, out_csv, summary_csv, confusion_csv)
    print()
    print(f"Wrote {out_csv.relative_to(REPO)}")
    print(f"Wrote {summary_csv.relative_to(REPO)}")
    print(f"Wrote {confusion_csv.relative_to(REPO)}")
    expected = len(names) * len(args.models) * args.replicates
    if len(final) != expected:
        print(f"Expected {expected} rows in this output; found {len(final)}", file=sys.stderr)
        return 3
    if not args.smoke:
        repeats = final.groupby(["name_id", "model_id"]).size()
        if len(final) != 192 or set(final["model_id"]) != {"glm-5", "glm-4.5"}:
            print(
                "Full run validation failed: expected 192 rows for glm-5 and glm-4.5",
                file=sys.stderr,
            )
            return 3
        if not (repeats == 3).all():
            print(
                "Full run validation failed: each name-model pair must have 3 repeats",
                file=sys.stderr,
            )
            return 3
    if not final.empty:
        summary = pd.read_csv(summary_csv)
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
