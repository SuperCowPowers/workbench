# Base image: python:3.12.5-bookworm
FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Set PYTHONHASHSEED to ensure consistent hashing across processes
ENV PYTHONHASHSEED=0

# Define workbench version
ARG WORKBENCH_VERSION=0.8.155

# Install system packages and gunicorn in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor libnghttp2-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Dependency Layer: Install all dependencies of workbench[ui]
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}" && \
    pip uninstall -y workbench && \
    pip cache purge

# Copy app code and configuration
COPY . /app
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

# Install workbench[ui] in separate layer (dependencies already satisfied)
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}"

# Configure Nginx and Supervisor
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord"]