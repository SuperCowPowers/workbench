# Multi-stage build for dependency resolution
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc

# Copy and install requirements file
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Final runtime image
FROM python:3.12-slim

# Install required X11 libraries for RDKit Draw module
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxinerama1 \
    libfontconfig1 \
    libcairo2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install workbench only
RUN pip install --no-cache-dir workbench==0.8.242

# Add the script runner
COPY ml_pipeline_runner.py /app/ml_pipeline_runner.py
WORKDIR /app

ENTRYPOINT ["python", "/app/ml_pipeline_runner.py"]
