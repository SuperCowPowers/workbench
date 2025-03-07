# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm AS deps-stage

# Set workbench version at the top for easy updates
ARG WORKBENCH_VERSION=0.8.111

# Install pip-tools and generate dependency list (excluding workbench itself)
RUN pip install --no-cache-dir pip-tools && \
    echo "workbench[ui]==${WORKBENCH_VERSION}" > /tmp/requirements.txt && \
    pip-compile --no-header --no-emit-index-url --no-emit-trusted-host --no-annotate /tmp/requirements.txt -o /tmp/deps.txt && \
    grep -v "workbench==" /tmp/deps.txt > /tmp/deps-only.txt

# Final image
FROM python:3.12.5-bookworm

# Set workbench version again (ARGs don't persist across FROM statements)
ARG WORKBENCH_VERSION=0.8.111

WORKDIR /app
EXPOSE 8000

# Install system packages, clean up in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir gunicorn

# Copy and install dependency list from deps stage
COPY --from=deps-stage /tmp/deps-only.txt /tmp/deps-only.txt
RUN pip install --no-cache-dir -r /tmp/deps-only.txt

# Configure Nginx and Supervisor
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install workbench in a separate layer
RUN pip install --no-cache-dir workbench[ui]==${WORKBENCH_VERSION}

# Copy app code and configuration
COPY . /app
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

CMD ["/usr/bin/supervisord"]
