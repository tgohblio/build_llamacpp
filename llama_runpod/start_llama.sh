#!/bin/bash
# Start llama-server in the background, then exec the Runpod handler.
#
# Required env vars (set by the serverless endpoint):
#   MODEL            - target model, e.g. "ggml-org/Qwen3.8-27B-GGUF:Q4_K_M"
#   DRAFT_MODEL      - draft model, e.g. "incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M"
# Optional:
#   N_GPU_LAYERS     - default 99 (all)
#   CTX_SIZE         - default 8192
#   PARALLEL         - default 1
#   PORT             - default 8080
#   SPEC_TYPE        - default draft-dflash
#   SPEC_DRAFT_N_MAX - default 7

set -e

: "${MODEL:?ERROR: MODEL env var required, e.g. ggml-org/Qwen3.8-27B-GGUF:Q4_K_M}"
: "${DRAFT_MODEL:?ERROR: DRAFT_MODEL env var required, e.g. incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M}"

N_GPU_LAYERS=${N_GPU_LAYERS:-99}
CTX_SIZE=${CTX_SIZE:-8192}
PARALLEL=${PARALLEL:-1}
PORT=${PORT:-8080}
# SPEC_TYPE is optional. When unset or empty, --spec-type and
# --spec-draft-n-max are omitted from the llama-server invocation entirely.
SPEC_TYPE=${SPEC_TYPE:-}
SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX:-7}

# Cache dir for HF downloads (serverless provides ephemeral container disk)
export HF_HOME=${HF_HOME:-/runpod-volume/huggingface-cache}
mkdir -p "$HF_HOME"

echo "[start_llama] MODEL=${MODEL} DRAFT_MODEL=${DRAFT_MODEL}"
echo "[start_llama] N_GPU_LAYERS=${N_GPU_LAYERS} CTX_SIZE=${CTX_SIZE} PARALLEL=${PARALLEL} PORT=${PORT}"
echo "[start_llama] SPEC_TYPE='${SPEC_TYPE}' SPEC_DRAFT_N_MAX=${SPEC_DRAFT_N_MAX}"

# Build optional spec-decoding args; omit them entirely when SPEC_TYPE is unset.
SPEC_ARGS=()
if [ -n "${SPEC_TYPE}" ]; then
    SPEC_ARGS=(--spec-type "${SPEC_TYPE}" --spec-draft-n-max "${SPEC_DRAFT_N_MAX}")
fi

# Launch llama-server in the background. Logs go to /tmp/llama-server.log.
/app/llama-server \
    -hf "${MODEL}" \
    -hfd "${DRAFT_MODEL}" \
    "${SPEC_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl "${N_GPU_LAYERS}" \
    -c "${CTX_SIZE}" \
    --parallel "${PARALLEL}" \
    > /tmp/llama-server.log 2>&1 &

LLAMA_PID=$!
echo "[start_llama] llama-server pid=${LLAMA_PID}"

# Wait for llama-server to be healthy (poll /health).
# On serverless, the worker must be ready before the handler accepts jobs.
TIMEOUT=600   # 10 min for cold start + model load + HF download
elapsed=0
while [ $elapsed -lt $TIMEOUT ]; do
    if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
        echo "[start_llama] llama-server healthy after ${elapsed}s"
        break
    fi
    # Also fail fast if the server died
    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "[start_llama] llama-server died, dumping log:"
        cat /tmp/llama-server.log
        exit 1
    fi
    sleep 5
    elapsed=$((elapsed + 5))
done

if ! curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[start_llama] llama-server failed to become healthy within ${TIMEOUT}s"
    cat /tmp/llama-server.log
    exit 1
fi

echo "[start_llama] launching Runpod handler"
exec python3 -u /app/handler.py
