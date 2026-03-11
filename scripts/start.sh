#!/usr/bin/env bash
# Start the FastAPI API and Flask UI with hot reload.
#
# Usage:
#   ./scripts/start.sh          # both services
#   ./scripts/start.sh api      # API only
#   ./scripts/start.sh ui       # UI only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../backend" && pwd)"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
UI_HOST="${UI_HOST:-127.0.0.1}"
UI_PORT="${UI_PORT:-5001}"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT INT TERM

start_api() {
    echo "Starting FastAPI on http://${API_HOST}:${API_PORT} (reload enabled)"
    cd "$PROJECT_DIR"
    uv run uvicorn src.api.app:create_app \
        --factory \
        --host "$API_HOST" \
        --port "$API_PORT" \
        --reload \
        --reload-dir src &
    PIDS+=($!)
}

start_ui() {
    echo "Starting Flask UI on http://${UI_HOST}:${UI_PORT} (reload enabled)"
    cd "$PROJECT_DIR"
    FLASK_DEBUG=1 uv run red-iron-square-ui &
    PIDS+=($!)
}

MODE="${1:-all}"

case "$MODE" in
    api)
        start_api
        ;;
    ui)
        start_ui
        ;;
    all|"")
        start_api
        start_ui
        ;;
    *)
        echo "Usage: $0 [api|ui|all]"
        exit 1
        ;;
esac

echo ""
echo "Press Ctrl+C to stop all services."
wait
