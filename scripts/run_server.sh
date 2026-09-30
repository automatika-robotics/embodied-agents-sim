#!/usr/bin/env bash
# Start the LeRobot async policy server in the foreground.
# Refuses to start if the port already has a listener: gRPC lets a second
# server bind to it silently and then splits requests between the two.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIM_ROOT="${SIM_ROOT:-$(dirname "$HERE")}"
PORT="${PORT:-8080}"

n=$(ss -tln 2>/dev/null | grep -c ":$PORT " || true)
if [ "${n:-0}" -gt 0 ]; then
    echo "ERROR: port $PORT already has a listener."
    echo "Stale policy server? Stop it first:  pkill -9 -f lerobot.async_inference.policy_server"
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Starting LeRobot policy server on 127.0.0.1:$PORT (no policy loaded until a client connects)"
exec "$SIM_ROOT/lerobot-venv/bin/python" -m lerobot.async_inference.policy_server \
    --host=127.0.0.1 --port="$PORT"
