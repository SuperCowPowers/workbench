# syntax=docker/dockerfile:1

# Multi-stage build for the chemprop inference image. Layer 1 installs all
# locked deps from requirements.lock (every transitive pinned, including
# torch==X.Y.Z+cpu from the PyTorch CPU index). Chemprop inference on small
# batches is CPU-bound; GPU buys no real speedup at low batch sizes. Layer 2
# installs workbench with --no-deps for small per-version-bump deltas.
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked deps (every transitive pinned). torch+cpu comes from
# the PyTorch CPU index — --extra-index-url + --index-strategy
# unsafe-best-match tell uv which index hosts the +cpu wheel.
COPY pytorch_chem/inference/requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

# Final runtime image
FROM python:3.12-slim

# Runtime system deps. The libX*/libexpat packages are for cuik-molmaker-pin, which chemprop
# imports unconditionally (chemprop/featurizers/molgraph/molecule.py). Its bundled cairo and
# fontconfig link against the X11 client libs and expat, none of which the slim base carries,
# and the builder stage copies site-packages across but not system libraries — so without
# these `import chemprop` dies at container start with "cannot open shared object file".
# Verified against the base image: ldd reports no unresolved libraries with this set.
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim libxrender1 libxext6 libsm6 libexpat1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy heavy deps from builder (changes rarely)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Layer 2: Workbench — rebuilds per version bump, ~20 MB delta.
ARG WORKBENCH_VERSION=0.8.467
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
