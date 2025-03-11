# Declare global argument for workbench version
ARG WORKBENCH_VERSION=0.8.116

# Base image using Python 3.12 and Debian Bookworm
FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Set PYTHONHASHSEED to ensure consistent hashing across processes
ENV PYTHONHASHSEED=0

# Install system packages and gunicorn in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor libnghttp2-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Copy and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy app code and configuration
COPY . /app
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

# Install workbench[ui] separately to keep the layer small and cacheable
ARG WORKBENCH_VERSION
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}"

# Configure Nginx and Supervisor
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
