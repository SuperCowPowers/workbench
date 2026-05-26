# Multi-stage build for the chemprop inference image. Layer 1 installs all
# deps in one pass — pip uses PyTorch's cu130 index for torch + chemprop
# and falls back to pypi (via --extra-index-url) for everything else.
# Layer 2 installs workbench with --no-deps for small per-version-bump deltas.
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Layer 1: All deps in one pass. torch + chemprop come from the cu130
# wheel index; the rest fall back to pypi via --extra-index-url.
COPY constraints.txt /tmp/
COPY pytorch_chem/inference/requirements.txt /tmp/
RUN pip install --no-cache-dir -c /tmp/constraints.txt \
    --index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple/ \
    -r /tmp/requirements.txt

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
# deployed-endpoint model bundles.
ARG WORKBENCH_VERSION=0.8.347
RUN pip install --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-cache-dir --no-deps "workbench-bridges==0.2.10"

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
