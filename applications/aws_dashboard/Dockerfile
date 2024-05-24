# Use Python 3.10
FROM python:3.10.13

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

# Install app-specific dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Nginx and Supervisor configuration files
COPY nginx.conf /etc/nginx/sites-available/default
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install Sageworks (changes often)
RUN pip install --no-cache-dir sageworks==0.6.6

# Copy the current directory contents into the container at /app
COPY . /app

# Grab the config file from build args, copy, and set ENV var
ARG SAGEWORKS_CONFIG
COPY $SAGEWORKS_CONFIG /app/sageworks_config.json
ENV SAGEWORKS_CONFIG=/app/sageworks_config.json

# Run supervisord
CMD ["/usr/bin/supervisord"]

