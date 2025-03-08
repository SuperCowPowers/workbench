# Base image stage
FROM python:3.12.5-bookworm AS base
ARG WORKBENCH_VERSION=0.8.112
WORKDIR /app
EXPOSE 8000

# Resolve and freeze dependencies (excluding workbench)
FROM base AS deps-stage
ARG WORKBENCH_VERSION=0.8.112
RUN pip install --no-cache-dir workbench[ui]==${WORKBENCH_VERSION} && \
    pip uninstall -y workbench && \
    pip freeze > /tmp/deps.txt

# Final image
FROM base
ARG WORKBENCH_VERSION=0.8.112

# Install system packages and gunicorn
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Install frozen dependencies
COPY --from=deps-stage /tmp/deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt

# Configure Nginx and Supervisor
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install workbench[ui] separately
RUN pip install --no-cache-dir workbench[ui]==${WORKBENCH_VERSION}

# Copy app code and configuration
COPY . /app
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

CMD ["/usr/bin/supervisord"]