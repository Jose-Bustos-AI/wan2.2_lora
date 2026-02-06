FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git wget curl jq \
    ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r /app/requirements.txt

# Install ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui && \
    pip3 install -r /comfyui/requirements.txt

# Optional: ComfyUI Manager (Good to have)
RUN git clone https://github.com/Comfy-Org/ComfyUI-Manager.git /comfyui/custom_nodes/ComfyUI-Manager || true

# Install missing custom nodes (Hunyuan Latent & Image Saver)
RUN git clone https://github.com/ShmuelRonen/ComfyUI-EmptyHunyuanLatent.git /comfyui/custom_nodes/ComfyUI-EmptyHunyuanLatent && \
    git clone https://github.com/giriss/comfy-image-saver.git /comfyui/custom_nodes/comfy-image-saver

# Copy app files
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
COPY workflows /app/workflows

# Permissions
RUN chmod +x /app/entrypoint.sh

# Default envs
ENV COMFY_HOST=127.0.0.1:8188
ENV COMFY_ROOT=/comfyui
ENV WORKFLOW_PATH=/app/workflows/workflow.json

CMD ["/app/entrypoint.sh"]
