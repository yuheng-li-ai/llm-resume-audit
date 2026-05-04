#!/usr/bin/env bash
# Phase 5.5 main batch — 8000 cells x 2 models = 16000 calls on Zhipu paid API.
#
# Two-model panel (locked at v0.5b-panel-trim):
#   - glm-5     (GLM 5.1, ~RMB 195 estimate)
#   - glm-4.5   (GLM 4.5, ~RMB 73 estimate from 50-cell diagnostic)
#
# Single-vendor (Zhipu) → no VPN needed; SDK hits api.bigmodel.cn directly.
# Wall-clock estimate: ~9 hours at RPM=30.
# Combined cost estimate: ~RMB 268 of RMB 300 balance (RMB 32 headroom).
#
# Resumable: re-running this script after a crash continues from where it
# stopped — BatchRunner skips any (cell_id, model_id) pair already in
# data/audit/scores.parquet.
#
# Usage (from repo root):
#     bash scripts/run_main_batch.sh                # run all 8000 cells
#     bash scripts/run_main_batch.sh --limit 100    # smoke-test 100 cells (200 calls, ~7 min)
#     bash scripts/run_main_batch.sh --chunk-size 500
#
# Output:
#     /tmp/main_batch.log                            # tee'd console
#     data/audit/scores.parquet                      # all results
#     data/audit/cost_log.csv                        # per-batch cost log

set -euo pipefail

# Strip any inherited HTTP(S)_PROXY (e.g. from a VPN client's local proxy on
# 127.0.0.1:10080). Zhipu's open.bigmodel.cn resolves to a China-mainland
# Aliyun IP and is reachable directly without any proxy; routing through a
# dead local proxy after the VPN closes manifests as "Connection error".
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy
export NO_PROXY="open.bigmodel.cn,api.bigmodel.cn,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

cd "$(dirname "$0")/.."
echo "Repo: $(pwd -P)"

if ! python -c "import llm_audit" 2>/dev/null; then
    echo "ERROR: 'llm_audit' not importable. Activate the conda env first:"
    echo "    conda activate llm-audit"
    exit 1
fi

for f in data/processed/treatment_assignments.parquet \
         data/processed/job_descriptions.parquet \
         data/processed/base_resumes.parquet; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required input: $f"
        exit 1
    fi
done

if [ ! -f .env ]; then
    echo "ERROR: .env not found at repo root"
    exit 1
fi

if ! grep -q '^ZHIPUAI_API_KEY=..' .env; then
    echo "ERROR: ZHIPUAI_API_KEY not populated in .env"
    exit 1
fi

LOG=/tmp/main_batch.log
echo "==========================================================="
echo "Phase 5.5 main batch → $LOG"
echo "Panel: glm-5 + glm-4.5 (Zhipu paid, single-vendor)"
echo "Cost estimate: ~RMB 268 | Wall-clock: ~9 hours at RPM=30"
echo "Resumable: re-run after crash to continue."
echo
echo "Reminder: CHECKLIST item 6.4 (OSF pre-registration lock) is the"
echo "intended hard gate before this batch. Verify before launching the"
echo "full 8000-cell run; --limit smoke-tests are fine pre-OSF."
echo "==========================================================="

python -m scripts._main_batch "$@" 2>&1 | tee "$LOG"

echo
echo "==========================================================="
echo "Done. Check:"
echo "  data/audit/scores.parquet"
echo "  data/audit/cost_log.csv"
echo "==========================================================="
