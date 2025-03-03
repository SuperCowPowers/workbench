FROM python:3.12-slim

# Install Vim
RUN apt-get update && apt-get install -y vim

# Copy requirements file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

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