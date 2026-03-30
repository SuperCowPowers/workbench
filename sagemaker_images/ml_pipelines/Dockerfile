# Multi-stage build for dependency resolution
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc

# Layer 1: All heavy dependencies (changes only when requirements.txt or pyproject.toml changes)
# Excludes workbench-bridges since it's installed alongside workbench in Layer 2
COPY ml_pipelines/requirements.txt /tmp/
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c "\
import tomllib; \
data = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
deps = [d for d in data['project']['dependencies'] if not d.startswith('workbench-bridges')]; \
open('/tmp/workbench_deps.txt', 'w').write('\n'.join(deps))" && \
    pip install --no-cache-dir -r /tmp/workbench_deps.txt && \
    rm /tmp/pyproject.toml /tmp/workbench_deps.txt

# Final runtime image
FROM python:3.12-slim

# System packages (single apt-get update)
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
ARG WORKBENCH_VERSION=0.8.296
RUN pip install --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-cache-dir --no-deps workbench-bridges

# Layer 3: App code (tiny, changes rarely)
COPY ml_pipelines/ml_pipeline_runner.py /app/ml_pipeline_runner.py
WORKDIR /app

ENTRYPOINT ["python", "/app/ml_pipeline_runner.py"]
