# syntax=docker/dockerfile:1

# GPU-capable training image. Lightweight CUDA base with Python 3.12.
# Uses the runtime image (not devel) to minimize size while supporting GPU
# inference/training. Single-stage build because the CUDA base image doesn't
# play nicely with the python:3.12-slim multi-stage pattern.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install pip and build tools (Python 3.12 is included in Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# GPU containers on Ubuntu use system Python (no venv) and need
# --break-system-packages to install into the externally-managed Python.
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked deps (every transitive pinned). torch+cu130 comes from
# the PyTorch CUDA index — --extra-index-url + --index-strategy
# unsafe-best-match tell uv which index hosts the +cu130 wheel.
COPY pytorch_chem/training/requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20 MB delta.
# workbench-bridges==0.2.10 retained for backward compat with existing
# deployed-endpoint model bundles.
ARG WORKBENCH_VERSION=0.8.388
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"
