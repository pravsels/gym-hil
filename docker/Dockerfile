ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libegl1 \
    libgl1 \
    libglib2.0-0 \
    libglfw3 \
    libhidapi-hidraw0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxkbcommon0 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    "lerobot>=0.4,<0.5" \
    av \
    datasets \
    einops \
    "gymnasium>=0.29.1,<2.0.0" \
    hidapi \
    "imageio[ffmpeg]>=2.34.0" \
    huggingface_hub \
    mujoco \
    pandas \
    pygame \
    pynput \
    pyyaml \
    safetensors \
    scipy \
    termcolor \
    wandb

# Reinstall torch with CUDA after lerobot — lerobot pulls CPU-only torch from
# PyPI on aarch64. The cu126 index is the earliest with aarch64 CUDA wheels.
RUN python -m pip install --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.6.0 torchvision==0.21.0

WORKDIR /workspace
