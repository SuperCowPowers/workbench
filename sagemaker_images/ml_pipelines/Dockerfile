# syntax=docker/dockerfile:1

# Multi-stage build for the headless pipeline runner. Layer 1a pre-installs
# CPU-only torch (claiming the torch slot before sagemaker-serve's
# transitive resolution would grab the multi-GB CUDA wheel from pypi);
# Layer 1b installs the full workbench dep cone. Layer 2 installs workbench
# itself with --no-deps for small per-version-bump deltas.
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc

COPY constraints.txt /tmp/
COPY ml_pipelines/requirements.txt /tmp/

# Layer 1a: Pre-install CPU torch from PyTorch's index. requirements.txt
# also lists torch (by name) so the version pin in constraints.txt is
# centralized; pip sees torch already installed when it processes
# requirements.txt below and skips re-resolving it. The pip cache mount
# (shared with Layer 1b) keeps downloaded wheels between builds.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/

# Layer 1b: Full workbench runtime deps for the pipeline runner — runs
# arbitrary user code that touches the full workbench surface.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt -r /tmp/requirements.txt

# Final runtime image
FROM python:3.12-slim

# System packages
RUN apt-get update && apt-get upgrade -y openssl && \
    apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
        libsm6 \
        libxinerama1 \
        libfontconfig1 \
        libcairo2 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Layer 1: All dependencies from builder (cached, changes rarely)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Layer 2: Workbench + Bridges (changes often, small layer)
# workbench-bridges==0.2.10 pinned for reproducibility while we still depend
# on it; on track for deprecation once nothing references workbench_bridges.
ARG WORKBENCH_VERSION=0.8.350
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"

# Layer 3: App code (tiny, changes rarely)
COPY ml_pipelines/ml_pipeline_runner.py /app/ml_pipeline_runner.py
WORKDIR /app

ENTRYPOINT ["python", "/app/ml_pipeline_runner.py"]
