#!/usr/bin/env bash
# GLM 4.5 diagnostic — 50 cells, float schema, anchored prompt, temp=0.
#
# Tests whether GLM 4.5 (Zhipu paid, mid-cost) gives a continuous score
# distribution comparable to GLM 5. If yes, GLM 4.5 can replace the three
# coarse free-tier models (GLM-4 Flash, Gemini 2.5 Flash, Llama 3.3 70B)
# in the locked panel.
#
# Same 50-cell sample as prior diagnostics (random_state=99) so
# distributions are directly comparable.
#
# Cost: ~RMB 0.5 per run (50 calls × ~RMB 0.01).
#
# Usage (from repo root):
#     bash scripts/run_glm45_diagnostic.sh
#
# Output: /tmp/glm45_float_diag.log + data/audit/scores_glm45_float_diag.parquet

set -euo pipefail

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

LOG=/tmp/glm45_float_diag.log
echo "==========================================================="
echo "GLM 4.5 (Zhipu paid) diagnostic — 50 cells → $LOG"
echo "Pre-flight cost estimate ~RMB 0.5"
echo "==========================================================="
python -m scripts._float_diagnostic glm45 2>&1 | tee "$LOG"

echo
echo "==========================================================="
echo "Done. Compare against:"
echo "  GLM 5 (paid, integer pilot):    24 unique, std 11.10  CONTINUOUS"
echo "  GLM-4 Flash (free, integer):     3 unique, std  4.41  COARSE"
echo "  Gemini 2.5 Flash (free, int):    3 unique, std  0.78  COARSE"
echo "  Llama 3.3 70B (free, integer):   4 unique, std  4.27  COARSE"
echo "==========================================================="
