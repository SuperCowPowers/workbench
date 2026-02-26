# Stage 1: Install workbench with all dependencies to capture the full dependency set
FROM python:3.12.5-bookworm AS deps-builder
ARG WORKBENCH_VERSION=0.8.272
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}"
# Freeze everything except workbench itself into a requirements file
RUN pip freeze | grep -v "^workbench==" > /tmp/deps.txt

# Stage 2: Build the final image
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

# Layer 1: All dependencies (changes rarely)
COPY --from=deps-builder /tmp/deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir uvicorn asgiref -r /tmp/deps.txt && rm /tmp/deps.txt

# Layer 2: Workbench only (changes often)
ARG WORKBENCH_VERSION=0.8.272
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
