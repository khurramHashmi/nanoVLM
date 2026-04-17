#!/bin/bash
# Evaluate nanoVLM checkpoints on VLMEvalKit benchmarks.
#
# Usage:
#   bash eval/eval_vlmevalkit.sh                                  # defaults: nanoVLM-460M-8k, MMStar
#   bash eval/eval_vlmevalkit.sh nanoVLM-230M-8k MMStar MME OCRBench # custom model + benchmarks
#
# For LLM-judged benchmarks (MMBench, MMVet), set these env vars:
#   export JUDGE_BASE_URL=http://your-judge:8880/v1
#   export JUDGE_KEY=EMPTY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${1:-nanoVLM-460M-8k}"
shift 2>/dev/null || true

if [ $# -gt 0 ]; then
    BENCHMARKS="$@"
else
    BENCHMARKS="MMStar"
fi

echo "=== VLMEvalKit: ${MODEL} ==="
echo "Benchmarks: ${BENCHMARKS}"
echo "Repo root:  ${REPO_ROOT}"

JUDGE_ARGS=""
if [ -n "${JUDGE_BASE_URL:-}" ]; then
    JUDGE_ARGS="--judge-base-url ${JUDGE_BASE_URL} --judge-key ${JUDGE_KEY:-EMPTY}"
    echo "Judge:      ${JUDGE_BASE_URL}"
fi

cd "${REPO_ROOT}"

python VLMEvalKit/run.py \
    --data ${BENCHMARKS} \
    --model "${MODEL}" \
    ${JUDGE_ARGS} \
    --verbose
