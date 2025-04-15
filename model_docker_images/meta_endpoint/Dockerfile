FROM python:3.12-slim AS builder

# Copy requirements file
COPY requirements.txt /tmp/
# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.12-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install vim and clean up in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add the serve script
COPY serve /usr/local/bin/
RUN chmod +x /usr/local/bin/serve

# Copy the main.py/entrypoint script
COPY main.py /opt/program/
WORKDIR /opt/program

# Make port 8080 available for the web server
EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=TRUE

# SageMaker will look for this
CMD ["serve"]

# Required label for SageMaker pipeline models
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true