# syntax=docker/dockerfile:1

# Multi-stage build. Layer 1 installs all training runtime deps from
# requirements.lock (every transitive pinned). Layer 2 installs workbench
# with --no-deps for small per-version-bump deltas. This image is a strict
# subset of `pip install workbench` — no sagemaker SDK, no orchestration
# clients (training jobs run under SageMaker's orchestration, not as
# orchestrators themselves).
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked training runtime deps (every transitive pinned).
COPY base/training/requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.lock

# Final runtime image
FROM python:3.12-slim

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy heavy deps from builder (changes rarely)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20 MB delta.
# workbench-bridges==0.2.10 is retained for backward compatibility with
# existing deployed-endpoint model bundles; drop in a future image rev once
# no production endpoint references workbench_bridges.
ARG WORKBENCH_VERSION=0.8.381
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"
