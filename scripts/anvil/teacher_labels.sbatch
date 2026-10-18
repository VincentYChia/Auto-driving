#!/bin/bash
# Generate reasoning labels with a large teacher VLM served by vLLM on one 4×A100 node.
# TEMPLATE: check that your vLLM version supports the chosen model, and its licence terms.
#
#   TEACHER_MODEL=google/gemma-4-31b-it LIMIT=200 sbatch scripts/anvil/teacher_labels.sbatch   # pilot batch
#SBATCH --job-name=teacher-labels
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/scripts/anvil/common.sh"
activate_env "${TEACHER_CONDA_ENV:-vllm}"
MODEL="${TEACHER_MODEL:?set TEACHER_MODEL}"
PORT="${PORT:-8000}"
VIEWS="$TRAINING_DATA_ROOT/views/navtrain"
OUT="$TRAINING_DATA_ROOT/labels/navtrain_teacher.jsonl"
mkdir -p "$(dirname "$OUT")"

vllm serve "$MODEL" --tensor-parallel-size 4 --port "$PORT" --max-model-len 8192 &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 10; done

cd "$REPO_ROOT"
EXTRA=()
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
python -m data_pipeline.label_client --requests "$VIEWS/teacher_requests.jsonl" --output "$OUT" \
    --base-url "http://localhost:$PORT/v1" --model "$MODEL" --concurrency 16 ${EXTRA[@]+"${EXTRA[@]}"}
