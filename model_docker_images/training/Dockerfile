FROM python:3.12-slim

# Install Vim
RUN apt-get update && apt-get install -y vim

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