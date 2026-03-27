# Lightweight CUDA base image with Python 3.12 support
# Uses runtime image (not devel) to minimize size while supporting GPU inference/training
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install pip and build tools (Python 3.12 is included in Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# GPU containers on Ubuntu use system Python (no venv) and need --break-system-packages
ENV BREAK_SYSTEM_PACKAGES=1

# Copy requirements file
COPY pytorch_chem/training/requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt
