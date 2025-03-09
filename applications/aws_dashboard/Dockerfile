# Declare global argument for workbench version
ARG WORKBENCH_VERSION=0.8.114

# Base image stage using Python 3.12 and Debian Bookworm
FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Stage to resolve and freeze workbench[ui] dependencies (excluding workbench itself)
FROM base AS deps-stage
ARG WORKBENCH_VERSION

# Install only workbench's dependencies, but not the package itself
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}" && \
    pip freeze | grep -v "workbench" > /tmp/deps.txt

# Final image stage
FROM base
ARG WORKBENCH_VERSION

# Install system packages and gunicorn in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Copy and install frozen dependencies from the deps-stage
COPY --from=deps-stage /tmp/deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt

# Copy app code and configuration
COPY . /app
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

# Install workbench[ui] separately to keep the layer small and cacheable
RUN pip install --no-cache-dir "workbench[ui]==${WORKBENCH_VERSION}"

# Configure Nginx and Supervisor
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

CMD ["/usr/bin/supervisord"]
