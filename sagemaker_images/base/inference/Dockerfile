# syntax=docker/dockerfile:1

# Multi-stage build. Layer 1 installs all endpoint runtime deps from
# requirements.lock (every transitive pinned). Layer 2 installs workbench
# itself with --no-deps so per-version-bump deltas are ~20 MB instead of a
# full pip resolve. This image is a strict subset of `pip install workbench`
# — no sagemaker SDK, no orchestration clients, no UI.
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked endpoint runtime deps (every transitive pinned).
COPY base/inference/requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/requirements.lock

# Final runtime image
FROM python:3.12-slim

# Runtime system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy heavy deps from builder (changes rarely)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Layer 2: Workbench — rebuilds per version bump, ~20 MB delta.
ARG WORKBENCH_VERSION=0.8.468
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}"

# Add the shared serve script
COPY shared/serve /usr/local/bin/
RUN chmod +x /usr/local/bin/serve

# Copy the shared main.py/entrypoint script
COPY shared/main.py /opt/program/
WORKDIR /opt/program

# Make port 8080 available for the web server
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=TRUE

# SageMaker will look for this
CMD ["serve"]

# Required label for SageMaker pipeline models
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
