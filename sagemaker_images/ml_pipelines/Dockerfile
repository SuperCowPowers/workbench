# syntax=docker/dockerfile:1

# Multi-stage build for the headless pipeline runner. Layer 1 installs the
# full workbench dep cone from requirements.lock (every transitive pinned
# with hashes). Layer 2 installs workbench itself with --no-deps for small
# per-version-bump deltas.
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

COPY ml_pipelines/requirements.lock /tmp/

# Layer 1: Install all locked dependencies (every transitive pinned —
# no silent cascades on rebuild). torch comes from the PyTorch CPU index
# (lockfile pins torch==2.12.0+cpu); --extra-index-url +
# --index-strategy unsafe-best-match tell uv which index hosts the +cpu wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

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
ARG WORKBENCH_VERSION=0.8.368
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"

# Layer 3: App code (tiny, changes rarely)
COPY ml_pipelines/ml_pipeline_runner.py /app/ml_pipeline_runner.py
WORKDIR /app

ENTRYPOINT ["python", "/app/ml_pipeline_runner.py"]
