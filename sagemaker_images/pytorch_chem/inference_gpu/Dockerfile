# syntax=docker/dockerfile:1

# GPU inference image, for models whose encoder needs CUDA to serve in reasonable time
# (Atom-JEPA). Single-stage on the same CUDA runtime base as the training image; serves with
# the shared FastAPI main.py like the CPU inference images. On a CPU instance torch falls
# back to CPU, so the instance type decides GPU vs CPU.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# python3-dev carries Python.h, which Triton compiles its CUDA launcher against the first
# time any Triton kernel runs (torch's built-in Triton ops).
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu's system Python is externally managed; installs go into it directly
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install uv (single static binary; used to install from the lockfile)
RUN pip install uv

# Layer 1: All locked deps (every transitive pinned). torch+cu130 comes from
# the PyTorch CUDA index — --extra-index-url + --index-strategy
# unsafe-best-match tell uv which index hosts the +cu130 wheel.
COPY pytorch_chem/inference_gpu/requirements.lock /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

# Layer 2: Workbench — rebuilds per version bump, ~20 MB delta.
ARG WORKBENCH_VERSION=0.8.509
RUN pip install --no-deps "workbench==${WORKBENCH_VERSION}"

# Add the shared serve script
COPY shared/serve /usr/local/bin/
RUN chmod +x /usr/local/bin/serve

# Copy the shared main.py/entrypoint script
COPY shared/main.py /opt/program/
WORKDIR /opt/program

# Make port 8080 available for the web server
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=TRUE

# SageMaker will look for this
CMD ["serve"]

# Required label for SageMaker pipeline models
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
