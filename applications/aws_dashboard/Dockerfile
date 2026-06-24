# syntax=docker/dockerfile:1

FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Set PYTHONHASHSEED to ensure consistent hashing across processes
ENV PYTHONHASHSEED=0

# Install system packages
# Upgrade OpenSSL to get security patches (DSA-6113-1)
RUN apt-get update && apt-get upgrade -y openssl && \
    apt-get install -y --no-install-recommends libnghttp2-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked dependencies (every transitive pinned; lockfile pins
# torch==X.Y.Z+cpu directly so we don't need the old two-step torch dance).
# --extra-index-url + --index-strategy unsafe-best-match tell uv which index
# hosts the +cpu wheel.
COPY requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

# Layer 2: Workbench only (changes often, ~20 MB)
ARG WORKBENCH_VERSION=0.8.383
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}"

# Copy app code and configuration (after pip installs to avoid cache invalidation)
COPY app.py /app/
COPY pages/ /app/pages/
COPY static/ /app/static/
COPY assets/ /app/assets/
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

# Run Uvicorn with WSGI-to-ASGI wrapper for Flask/Dash
CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "8000"]
