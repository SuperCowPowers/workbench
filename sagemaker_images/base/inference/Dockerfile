# syntax=docker/dockerfile:1

# Multi-stage build. Layer 1 has the endpoint runtime deps; Layer 2 installs
# workbench itself with --no-deps so per-version-bump deltas are ~20 MB
# instead of a full pip resolve. Version pins are centralized in
# /constraints.txt (copied into the build context by build_deploy.sh).
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Layer 1: Endpoint runtime deps. requirements.txt lists what this image
# actually needs (no sagemaker SDK, no orchestration clients — the endpoint
# surface is a strict subset of what `pip install workbench` delivers).
# The pip cache mount keeps downloaded wheels between builds — rebuilding
# after a constraints.txt bump only re-installs, doesn't re-download.
COPY constraints.txt /tmp/
COPY base/inference/requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -c /tmp/constraints.txt -r /tmp/requirements.txt

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

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20 MB delta.
# workbench-bridges==0.2.10 retained for backward compat with existing
# deployed-endpoint model bundles; drop in a future image rev once no
# production endpoint references workbench_bridges.
ARG WORKBENCH_VERSION=0.8.350
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-deps "workbench-bridges==0.2.10"

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
