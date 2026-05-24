# GPU-capable training image. Lightweight CUDA base with Python 3.12.
# Uses runtime image (not devel) to minimize size while supporting GPU
# inference/training. Single-stage build because the CUDA base image doesn't
# play nicely with the python:3.12-slim multi-stage pattern; layer isolation
# still works via two separate RUN commands (heavy deps then workbench).
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Install pip and build tools (Python 3.12 is included in Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# GPU containers on Ubuntu use system Python (no venv) and need --break-system-packages
ENV BREAK_SYSTEM_PACKAGES=1

# Layer 1: Framework extras (torch from PyTorch's index, chemprop, shap) +
# workbench's transitive deps + [modeling] extras. Auto-extract from
# pyproject.toml so we don't maintain a duplicated dep list.
COPY pytorch_chem/training/requirements.txt /tmp/
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt && \
    python -c "\
import tomllib; \
data = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
deps = data['project']['dependencies'] + data['project']['optional-dependencies']['modeling']; \
open('/tmp/workbench_deps.txt', 'w').write('\n'.join(deps))" && \
    pip install --break-system-packages --no-cache-dir -r /tmp/workbench_deps.txt && \
    rm /tmp/pyproject.toml /tmp/workbench_deps.txt

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20MB delta.
# workbench-bridges==0.2.10 retained for backward compat with existing
# deployed-endpoint model bundles.
ARG WORKBENCH_VERSION=0.8.344
RUN pip install --break-system-packages --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --break-system-packages --no-cache-dir --no-deps "workbench-bridges==0.2.10"
