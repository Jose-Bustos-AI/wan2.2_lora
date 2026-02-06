#!/usr/bin/env bash
set -euo pipefail

export COMFY_HOST="${COMFY_HOST:-127.0.0.1:8188}"
export COMFY_ROOT="${COMFY_ROOT:-/comfyui}"
export WORKFLOW_PATH="${WORKFLOW_PATH:-/app/workflows/workflow.json}"

echo "[entrypoint] COMFY_HOST=$COMFY_HOST"
echo "[entrypoint] COMFY_ROOT=$COMFY_ROOT"
echo "[entrypoint] WORKFLOW_PATH=$WORKFLOW_PATH"

# Start ComfyUI
echo "[entrypoint] Starting ComfyUI..."
cd "$COMFY_ROOT"
python3 main.py --listen 0.0.0.0 --port 8188 > /var/log/comfyui.log 2>&1 &
COMFY_PID=$!

# Wait for ComfyUI to be ready
echo "[entrypoint] Waiting for ComfyUI to be ready..."
for i in $(seq 1 240); do
  if curl -s "http://${COMFY_HOST}/" >/dev/null 2>&1; then
    echo "[entrypoint] ComfyUI is ready."
    break
  fi
  sleep 1
  if ! kill -0 "$COMFY_PID" >/dev/null 2>&1; then
    echo "[entrypoint] ComfyUI process died. Dumping logs:"
    tail -n 200 /var/log/comfyui.log || true
    exit 1
  fi
  if [ "$i" -eq 240 ]; then
    echo "[entrypoint] Timeout waiting for ComfyUI. Dumping logs:"
    tail -n 200 /var/log/comfyui.log || true
    exit 1
  fi
done

# Start handler
echo "[entrypoint] Starting RunPod handler..."
cd /app
python3 -u handler.py
