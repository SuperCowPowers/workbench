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

# GPU containers on Ubuntu use system Python (no venv) and need --break-system-packages
ENV BREAK_SYSTEM_PACKAGES=1

# Layer 1: All deps in one pass. torch + chemprop come from the cu130 wheel
# index; the rest (incl. shap and the workbench python libs) fall back to
# pypi via --extra-index-url.
COPY constraints.txt /tmp/
COPY pytorch_chem/training/requirements.txt /tmp/
RUN pip install --break-system-packages --no-cache-dir \
    -c /tmp/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple/ \
    -r /tmp/requirements.txt

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20 MB delta.
# workbench-bridges==0.2.10 retained for backward compat with existing
# deployed-endpoint model bundles.
ARG WORKBENCH_VERSION=0.8.350
RUN pip install --break-system-packages --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --break-system-packages --no-cache-dir --no-deps "workbench-bridges==0.2.10"
