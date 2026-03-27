FROM python:3.12-slim

# Install system dependencies
# libgomp1 is required for LightGBM
RUN apt-get update && apt-get install -y \
    vim \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY base/training/requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt
