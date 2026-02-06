#!/bin/bash
set -e

echo "[entrypoint] Starting ComfyUI..."
python3 /comfyui/main.py --listen 0.0.0.0 --port 8188 &

echo "[entrypoint] Waiting for ComfyUI..."
for i in {1..120}; do
  if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
    echo "[entrypoint] ComfyUI is ready."
    break
  fi
  sleep 1
done

echo "[entrypoint] Starting RunPod handler..."
exec python3 /app/handler.py
