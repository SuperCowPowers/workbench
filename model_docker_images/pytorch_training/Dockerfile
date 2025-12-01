# Lightweight CUDA base image with Python 3.12 support
# Uses runtime image (not devel) to minimize size while supporting GPU inference/training
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

# Install pip and build tools (Python 3.12 is included in Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

# Copy the SageMaker entrypoint script
COPY sagemaker_entrypoint.py /opt/program/
WORKDIR /opt/program

# Make the entrypoint executable
RUN chmod +x /opt/program/sagemaker_entrypoint.py

# Set the entrypoint
ENTRYPOINT ["/opt/program/sagemaker_entrypoint.py"]