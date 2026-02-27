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

# Layer 1: All dependencies (changes only when pyproject.toml deps change)
# pyproject.toml is copied into build context by deploy.sh
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "\
import tomllib; \
data = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
deps = data['project']['dependencies'] + data['project']['optional-dependencies']['ui']; \
open('/tmp/deps.txt', 'w').write('\n'.join(deps))" && \
    pip install --no-cache-dir uvicorn asgiref -r /tmp/deps.txt && \
    rm /tmp/pyproject.toml /tmp/deps.txt

# Layer 2: Workbench only (changes often, ~20MB)
ARG WORKBENCH_VERSION=0.8.273
RUN pip install --no-cache-dir --no-deps "workbench[ui]==${WORKBENCH_VERSION}"

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
