# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm AS deps-stage

# Set workbench version at the top for easy updates
ARG WORKBENCH_VERSION=0.8.110

# Install pip-tools and generate dependency list (excluding workbench itself)
RUN pip install --no-cache-dir pip-tools && \
    echo "workbench==${WORKBENCH_VERSION}" > /tmp/requirements.txt && \
    pip-compile --no-header --no-emit-index-url --no-emit-trusted-host --no-annotate /tmp/requirements.txt -o /tmp/deps.txt && \
    grep -v "workbench==" /tmp/deps.txt > /tmp/deps-only.txt

# Final image
FROM python:3.12.5-bookworm

# Set workbench version again (ARGs don't persist across FROM statements)
ARG WORKBENCH_VERSION=0.8.110

# Update system packages and handle vulnerabilities in one step
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libexpat1 libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependency list from deps stage
COPY --from=deps-stage /tmp/deps-only.txt /tmp/deps-only.txt
RUN pip install --no-cache-dir -r /tmp/deps-only.txt

# Install workbench in a separate layer
RUN pip install --no-cache-dir workbench==${WORKBENCH_VERSION}
