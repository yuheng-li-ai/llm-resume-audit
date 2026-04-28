"""BatchRunner — orchestrates calls across cells × models with rate
limiting, retries, dedup, calibration injection, and cost logging.

Pilot scope (Phase 5.4): single client (GLM-4 Flash), 100 cells, no
calibration. Main-batch scope (Phase 5.5): all 4 model_ids × 8,000 cells
with weekly calibration runs.

Rate limiting: simple token-bucket per provider keyed on requests-per-
minute. Free-tier defaults are conservative; override per call.

Retry: tenacity with exponential backoff up to N attempts on transient
failures (network, 429, 5xx, JSON parse errors).

Dedup: if scores.parquet already exists at output_path, the runner skips
(cell_id, model_id) pairs already present.
"""

from __future__ import annotations

import csv
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_audit.scoring.base import ScoringClient, ScoringResult
from llm_audit.treatment_injector import TreatmentCell
from llm_audit.utils.prompts import ScoringPrompt

logger = logging.getLogger(__name__)

_RETRY_ON: tuple[type[BaseException], ...] = (Exception,)


def _now_iso_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class CostTracker:
    """Accumulate per-call token + cost rows; emit CSV log."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def record(self, result: ScoringResult, batch_id: str) -> None:
        self._rows.append(
            {
                "provider": result.provider,
                "model_id": result.model_id,
                "batch_id": batch_id,
                "cell_id": result.cell_id,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_local": result.cost_local,
                "currency": result.currency,
                "latency_ms": result.latency_ms,
                "timestamp": _now_iso_utc(),
            }
        )

    def total_cost(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self._rows:
            out[r["currency"]] = out.get(r["currency"], 0.0) + float(r["cost_local"])
        return out

    def total_tokens(self) -> dict[str, int]:
        return {
            "tokens_in": sum(int(r["tokens_in"]) for r in self._rows),
            "tokens_out": sum(int(r["tokens_out"]) for r in self._rows),
        }

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            return
        fieldnames = list(self._rows[0].keys())
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for row in self._rows:
                writer.writerow(row)


class _RateLimiter:
    """Trivial RPM rate limiter — enforces minimum interval between calls."""

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / max(rpm, 1)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()


class BatchRunner:
    """Score a list of TreatmentCells across one or more clients."""

    def __init__(
        self,
        clients: dict[str, ScoringClient],
        cost_tracker: CostTracker | None = None,
        rpm_per_provider: dict[str, int] | None = None,
        max_retries: int = 4,
    ) -> None:
        self._clients = clients
        self._cost_tracker = cost_tracker or CostTracker()
        self._rate_limiters: dict[str, _RateLimiter] = {
            name: _RateLimiter(rpm) for name, rpm in (rpm_per_provider or {}).items()
        }
        self._max_retries = max_retries

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    def run_with_prompts(
        self,
        prompts: list[tuple[TreatmentCell, ScoringPrompt]],
        model_ids: list[str],
        scores_output_path: Path,
        cost_log_path: Path | None = None,
    ) -> pd.DataFrame:
        """Score (cell, prompt) pairs against the given model_ids."""
        batch_id = uuid.uuid4().hex[:12]
        existing = self._load_existing_scores(scores_output_path)
        rows: list[dict[str, Any]] = []

        for cell, prompt in prompts:
            for model_id in model_ids:
                if (cell.cell_id, model_id) in existing:
                    continue
                client = self._clients[model_id]
                row = self._score_one(client, cell, prompt, batch_id, model_id)
                rows.append(row)

        df = pd.DataFrame(rows)
        if existing and scores_output_path.exists():
            old = pd.read_parquet(scores_output_path)
            df = pd.concat([old, df], ignore_index=True)
        scores_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(scores_output_path, index=False)

        if cost_log_path is not None:
            self._cost_tracker.write_csv(cost_log_path)

        return df

    def _score_one(
        self,
        client: ScoringClient,
        cell: TreatmentCell,
        prompt: ScoringPrompt,
        batch_id: str,
        model_id: str,
    ) -> dict[str, Any]:
        if client.provider in self._rate_limiters:
            self._rate_limiters[client.provider].wait()

        @retry(  # type: ignore[misc]
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=30),
            retry=retry_if_exception_type(_RETRY_ON),
            reraise=True,
        )
        def call() -> ScoringResult:
            return client.score(prompt, cell_id=cell.cell_id)

        try:
            result = call()
            self._cost_tracker.record(result, batch_id)
            return {
                "cell_id": result.cell_id,
                "model_id": result.model_id,
                "provider": result.provider,
                "hiring_score": result.hiring_score,
                "rationale": result.rationale,
                "latency_ms": result.latency_ms,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_local": result.cost_local,
                "currency": result.currency,
                "batch_id": batch_id,
                "retrieved_at": _now_iso_utc(),
                "error": None,
            }
        except (RetryError, Exception) as exc:
            logger.warning(
                "score failed cell=%s model=%s err=%s",
                cell.cell_id,
                model_id,
                exc,
            )
            return {
                "cell_id": cell.cell_id,
                "model_id": model_id,
                "provider": client.provider,
                "hiring_score": None,
                "rationale": None,
                "latency_ms": None,
                "tokens_in": None,
                "tokens_out": None,
                "cost_local": None,
                "currency": None,
                "batch_id": batch_id,
                "retrieved_at": _now_iso_utc(),
                "error": str(exc)[:500],
            }

    @staticmethod
    def _load_existing_scores(path: Path) -> set[tuple[int, str]]:
        if not path.exists():
            return set()
        df = pd.read_parquet(path)
        return {(int(r["cell_id"]), str(r["model_id"])) for _, r in df.iterrows()}
