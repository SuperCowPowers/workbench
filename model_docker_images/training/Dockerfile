FROM python:3.12-slim

# Install system dependencies
# libgomp1 is required for LightGBM
RUN apt-get update && apt-get install -y \
    vim \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the SageMaker entrypoint script
COPY sagemaker_entrypoint.py /opt/program/
WORKDIR /opt/program

# Make the entrypoint executable
RUN chmod +x /opt/program/sagemaker_entrypoint.py

# Set the entrypoint
ENTRYPOINT ["/opt/program/sagemaker_entrypoint.py"]