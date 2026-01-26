# Define workbench version once at the top
ARG WORKBENCH_VERSION=0.8.237

# Stage 1: Extract dependencies from workbench[ui]
FROM python:3.12.5-bookworm AS deps-extractor
ARG WORKBENCH_VERSION
RUN pip install --no-cache-dir pip-tools && \
    pip download --no-deps "workbench[ui]==${WORKBENCH_VERSION}" -d /tmp/pkg && \
    cd /tmp/pkg && unzip -q *.whl -d /tmp/extracted && \
    cat /tmp/extracted/*.dist-info/METADATA | grep "^Requires-Dist:" | \
    sed 's/Requires-Dist: //' | sed 's/;.*//' | sed 's/ (.*//' > /tmp/deps.txt

# Stage 2: Build the final image
FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Set PYTHONHASHSEED to ensure consistent hashing across processes
ENV PYTHONHASHSEED=0

# Install system packages and uvicorn in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnghttp2-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uvicorn asgiref

# Copy and install dependencies (this layer only changes when deps change)
COPY --from=deps-extractor /tmp/deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt && rm /tmp/deps.txt

# Install workbench[ui] in separate layer (dependencies already satisfied)
ARG WORKBENCH_VERSION
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
