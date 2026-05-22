# Multi-stage build for dependency layer isolation. Layer 1 has all the heavy
# pip deps (torch from PyTorch's index, chemprop, fastapi/uvicorn, plus
# workbench's transitive deps auto-extracted from pyproject.toml). Layer 2
# installs workbench itself with --no-deps so per-version-bump deltas are
# small (~20MB instead of full pip resolve).
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Layer 1: Framework extras (torch from PyTorch's index, chemprop, fastapi,
# uvicorn) + workbench's transitive deps (bare, no [modeling] — chemprop
# endpoints don't import xgboost or mordred at runtime). Auto-extract from
# pyproject.toml so we don't maintain a duplicated dep list.
COPY pytorch_chem/inference/requirements.txt /tmp/
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c "\
import tomllib; \
data = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
deps = data['project']['dependencies']; \
open('/tmp/workbench_deps.txt', 'w').write('\n'.join(deps))" && \
    pip install --no-cache-dir -r /tmp/workbench_deps.txt && \
    rm /tmp/pyproject.toml /tmp/workbench_deps.txt

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

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20MB delta.
# workbench-bridges==0.2.10 retained for backward compat with existing
# deployed-endpoint model bundles.
ARG WORKBENCH_VERSION=0.8.342
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
