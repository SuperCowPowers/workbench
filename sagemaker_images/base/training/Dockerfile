# syntax=docker/dockerfile:1

# Multi-stage build. Layer 1 has the training runtime deps; Layer 2
# installs workbench with --no-deps for small per-version-bump deltas.
# Version pins are centralized in /constraints.txt (copied into the build
# context by build_deploy.sh).
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Layer 1: Training runtime deps. requirements.txt lists the workbench
# python libs this image needs (no orchestration SDKs — training jobs run
# under SageMaker's orchestration, not as orchestrators themselves).
# The pip cache mount keeps downloaded wheels between builds — rebuilding
# after a constraints.txt bump only re-installs, doesn't re-download.
COPY constraints.txt /tmp/
COPY base/training/requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt -r /tmp/requirements.txt

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
ARG WORKBENCH_VERSION=0.8.351
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"
