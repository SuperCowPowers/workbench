# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm

# Set the working directory
WORKDIR /app

# Install Nginx, Supervisor, and other necessary packages while removing vulnerabilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Gunicorn
RUN pip install --no-cache-dir gunicorn

# Inform Docker that the container is listening on port 8000
EXPOSE 8000

# Copy the requirements and install dependencies (except Workbench)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install Workbench Dashboard on its own layer
RUN pip install --no-cache-dir workbench==0.8.90

# Copy the Nginx and Supervisor configuration files
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy the current directory contents into the container at /app
COPY . /app

# Grab the config file from build args, copy, and set ENV var
ARG WORKBENCH_CONFIG
COPY $WORKBENCH_CONFIG /app/workbench_config.json
ENV WORKBENCH_CONFIG=/app/workbench_config.json

# Run supervisord
CMD ["/usr/bin/supervisord"]
