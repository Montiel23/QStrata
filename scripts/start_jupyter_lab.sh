#!/usr/bin/env bash
# scripts/start_jupyter_lab.sh
#
# Launch JupyterLab inside the qstrata-gpu Docker container.
# Run from the repo root:
#   bash scripts/start_jupyter_lab.sh
#
# JupyterLab is configured as the container's default command in
# infra/docker/docker-compose.gpu.yml and starts automatically
# when the container is brought up.
#
# Host port: 8889 → container port 8888
# No token required (--IdentityProvider.token='' is set in compose).
# After launch, open: http://localhost:8889

set -e

COMPOSE_FILE="infra/docker/docker-compose.gpu.yml"

# Resolve repo root regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=== QStrata — JupyterLab Launcher ==="
echo "Compose file : ${COMPOSE_FILE}"
echo ""

# Check if qstrata-gpu container is already running
RUNNING=$(docker compose -f "${COMPOSE_FILE}" ps --status running --services 2>/dev/null | grep -c "qstrata-gpu" || true)

if [ "${RUNNING}" -gt 0 ]; then
    echo "Container qstrata-gpu is already running."
    echo ""
else
    echo "Starting qstrata-gpu container (JupyterLab will start automatically)..."
    docker compose -f "${COMPOSE_FILE}" up -d qstrata-gpu
    echo ""
    echo "Waiting for JupyterLab to initialise..."
    sleep 5
fi

echo "=== Access ==="
echo ""
echo "  URL  : http://localhost:8889"
echo "  Auth : No token required"
echo ""
echo "  Notebooks    : /workspace/notebooks/"
echo "  Dataset      : /datasets/vindr-spinexr  (read-only)"
echo ""
echo "To stop: docker compose -f ${COMPOSE_FILE} down"
