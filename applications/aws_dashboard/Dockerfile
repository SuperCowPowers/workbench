# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm

# Set the working directory
WORKDIR /app

# Install Nginx and Supervisor, and clean up afterwards
RUN apt-get update \
    && apt-get install -y nginx supervisor \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Gunicorn
RUN pip install --no-cache-dir gunicorn

# Inform Docker that the container is listening on port 8000
EXPOSE 8000

# Remove git from the base image (vulnerability)
RUN apt-get remove --purge -y git && apt-get autoremove -y && apt-get clean

# Remove AOM from the base image (vulnerability)
RUN apt-get remove --purge -y libaom3 && apt-get autoremove -y && apt-get clean

# Upgrade the nghttp2 package to fix a vulnerability
RUN apt-get update && apt-get install -y libnghttp2-dev && apt-get clean

# Install SageWorks Dashboard dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Nginx and Supervisor configuration files
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install latest Sageworks
RUN pip install --no-cache-dir sageworks==0.8.11

# Remove setuptools (vulnerability)
# Reinstall supervisor after removing setuptools and pkg_resources
RUN apt-get remove --purge -y python3-setuptools python3-pkg-resources && \
    apt-get install -y supervisor

# Remove pip (vulnerability)
RUN python -m pip uninstall -y pip && \
    rm -rf /usr/local/lib/python*/dist-packages/pip /usr/local/bin/pip* && \
    apt-get autoremove -y && apt-get clean

# Copy the current directory contents into the container at /app
COPY . /app

# Grab the config file from build args, copy, and set ENV var
ARG SAGEWORKS_CONFIG
COPY $SAGEWORKS_CONFIG /app/sageworks_config.json
ENV SAGEWORKS_CONFIG=/app/sageworks_config.json

# Run supervisord
CMD ["/usr/bin/supervisord"]

