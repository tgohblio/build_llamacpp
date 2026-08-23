#!/usr/bin/env bash
# Entrypoint for the Runpod self-hosted runner image.
# Registers the GitHub Actions runner (if a token was supplied) and runs it.
set -euo pipefail
cd /runner

# Register only when a token is provided, and only if not already configured
# (so a paused/restarted pod does not try to re-register and fail).
if [ ! -f .runner ] && [ -n "${RUNNER_TOKEN:-}" ] && [ -n "${REPO_URL:-}" ]; then
  ./config.sh \
    --url "${REPO_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME:-runner}" \
    --labels "${LABELS:-self-hosted}" \
    --unattended \
    ${EPHEMERAL:+--ephemeral}
fi

exec ./run.sh
