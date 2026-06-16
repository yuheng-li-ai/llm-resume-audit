#!/usr/bin/env bash
# Phase 8 robustness runner.
#
# Default mode runs:
#   8.1 permutation placebo       local only
#   8.2 Borda ranking robustness  Zhipu API only, direct connection, resumable
#   8.3 temporal drift check      local only
#
# No OpenAI endpoint or SDK is used. The only networked step is 8.2, which
# calls Zhipu/BigModel directly. Proxy variables are cleared here, and
# scripts/run_phase8_borda.py also uses httpx trust_env=False.
#
# Usage from repo root:
#   bash scripts/run_phase8_direct_zhipu.sh
#   bash scripts/run_phase8_direct_zhipu.sh local   # 8.1 + 8.3 only, no network
#   bash scripts/run_phase8_direct_zhipu.sh borda   # 8.2 only, Zhipu direct
#   bash scripts/run_phase8_direct_zhipu.sh smoke   # one cheap direct Zhipu test call

set -euo pipefail

MODE="${1:-all}"

cd "$(dirname "$0")/.."
echo "Repo: $(pwd -P)"

# Force direct networking. This prevents dead VPN/local-proxy settings from
# leaking into subprocesses.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
unset ALL_PROXY all_proxy SOCKS_PROXY socks_proxy
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_ORG_ID OPENAI_PROJECT
export NO_PROXY="api.bigmodel.cn,open.bigmodel.cn,bigmodel.cn,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

if ! python -c "import llm_audit" 2>/dev/null; then
    echo "ERROR: 'llm_audit' not importable. Activate the project env first:"
    echo "    python -m pip install -e ."
    echo "or activate the conda/venv environment used for this project."
    exit 1
fi

for f in data/audit/scores.parquet \
         data/processed/treatment_assignments.parquet \
         data/processed/job_descriptions.parquet; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing required input: $f"
        exit 1
    fi
done

require_zhipu_key() {
    if [ ! -f .env ]; then
        echo "ERROR: .env not found at repo root"
        exit 1
    fi
    if ! grep -q '^ZHIPUAI_API_KEY=..' .env; then
        echo "ERROR: ZHIPUAI_API_KEY not populated in .env"
        exit 1
    fi
}

run_local() {
    echo
    echo "== Phase 8.1: permutation placebo =="
    python -u -m scripts.run_phase8_placebo

    echo
    echo "== Phase 8.3: temporal drift check =="
    python -u -m scripts.run_phase8_drift
}

run_borda() {
    require_zhipu_key
    echo
    echo "== Phase 8.2: Borda ranking robustness =="
    echo "Network: Zhipu/BigModel direct only; proxy env cleared; OpenAI env cleared."
    echo "Resume file: data/audit/borda_rankings.parquet"
    python -u -m scripts.run_phase8_borda
}

run_smoke() {
    require_zhipu_key
    echo
    echo "== Direct Zhipu smoke test =="
    python -u - <<'PY'
import os
import time

import httpx
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv(".env")
client = ZhipuAI(
    api_key=os.environ["ZHIPUAI_API_KEY"],
    timeout=30.0,
    http_client=httpx.Client(timeout=30.0, trust_env=False),
)
t0 = time.time()
resp = client.chat.completions.create(
    model="glm-5",
    messages=[
        {"role": "system", "content": "Reply with exactly: ok"},
        {"role": "user", "content": "direct zhipu connectivity test"},
    ],
    temperature=0.0,
    max_tokens=8,
    timeout=30.0,
)
print(f"OK {time.time() - t0:.1f}s: {resp.choices[0].message.content}")
PY
}

case "$MODE" in
    all)
        run_local
        run_borda
        ;;
    local)
        run_local
        ;;
    borda)
        run_borda
        ;;
    smoke)
        run_smoke
        ;;
    *)
        echo "ERROR: unknown mode '$MODE'. Use: all | local | borda | smoke"
        exit 2
        ;;
esac

echo
echo "== Phase 8 deferred items =="
echo "8.4 open-weight vs commercial gap: deferred/infeasible under locked single-vendor panel."
echo "8.5 photo extension: deferred; requires Phase 2b face assets + multimodal path."
echo
echo "Expected outputs:"
echo "  outputs/tables/placebo_summary.csv"
echo "  outputs/figures/placebo.pdf"
echo "  data/audit/borda_rankings.parquet"
echo "  outputs/tables/borda_comparison.csv"
echo "  outputs/figures/borda_comparison.pdf"
echo "  outputs/tables/drift_temporal_split.csv"
echo "  outputs/figures/drift_temporal.pdf"
