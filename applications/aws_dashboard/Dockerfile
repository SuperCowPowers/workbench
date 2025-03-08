# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm AS deps-stage

# Set workbench version at the top for easy updates
ARG WORKBENCH_VERSION=0.8.112

# Install pip-tools and generate dependency list (excluding workbench itself)
RUN pip install --no-cache-dir pip-tools && \
    # Create a requirements.in file for pip-compile
    echo "# Do not include workbench itself here" > /tmp/requirements.in && \
    # Generate the dependencies of workbench[ui] without installing
    pip download --no-deps workbench[ui]==${WORKBENCH_VERSION} && \
    pip download --no-binary :all: --no-deps --dest /tmp --quiet workbench[ui]==${WORKBENCH_VERSION} && \
    pip install -q wheel && \
    cd /tmp && unzip -q workbench-*.whl "*.dist-info/METADATA" && \
    grep "Requires-Dist:" */METADATA | sed 's/.*Requires-Dist: \(.*\) .*/\1/' | \
    grep -v "extra ==" > /tmp/requirements.in && \
    # Compile exact dependencies
    pip-compile --no-header --no-emit-index-url --no-emit-trusted-host --no-annotate /tmp/requirements.in -o /tmp/deps.txt

# Final image
FROM python:3.12.5-bookworm

# Set workbench version again (ARGs don't persist across FROM statements)
ARG WORKBENCH_VERSION=0.8.112

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
COPY --from=deps-stage /tmp/deps.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt

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