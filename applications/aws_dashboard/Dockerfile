FROM python:3.12.5-bookworm AS base
WORKDIR /app
EXPOSE 8000

# Set PYTHONHASHSEED to ensure consistent hashing across processes
ENV PYTHONHASHSEED=0

# Install system packages
# Upgrade OpenSSL to get security patches (DSA-6113-1)
RUN apt-get update && apt-get upgrade -y openssl && \
    apt-get install -y --no-install-recommends libnghttp2-dev && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Layer 1a: CPU-only PyTorch first — sagemaker v3 requires torch via
# sagemaker-serve; the default linux wheel is the multi-GB CUDA build.
# Pre-installing the CPU variant claims the torch slot so the later deps
# resolve doesn't drag the CUDA wheel in.
COPY constraints.txt /tmp/
RUN pip install --no-cache-dir -c /tmp/constraints.txt torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple/

# Layer 1b: All other dependencies (changes only when requirements.txt or
# constraints.txt changes). requirements.txt is the full workbench surface
# + [ui] extras (dash/plotly/matplotlib) + uvicorn/asgiref for the server.
# torch is also listed there but already installed in Layer 1a — pip skips
# re-resolving it. deploy.sh copies constraints.txt into the build context.
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -c /tmp/constraints.txt -r /tmp/requirements.txt

# Layer 2: Workbench only (changes often, ~20 MB)
ARG WORKBENCH_VERSION=0.8.350
RUN pip install --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}"

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
