# Use Python 3.10
FROM python:3.10.13

# Install Nginx
RUN apt-get update && apt-get install -y nginx

# Inform Docker that the container is listening on port 8000
EXPOSE 8000

# Set the working directory
WORKDIR /app

# The Dash Application is using relative imports :(
ENV PYTHONPATH /app

# Install Dash, Gunicorn, and Sageworks
RUN pip install dash gunicorn sageworks

# Copy the Nginx configuration file
COPY nginx.conf /etc/nginx/sites-available/default

# Copy the current directory contents into the container at /app
COPY . /app

# Set the ENV var for the SageWorks config file
ENV SAGEWORKS_CONFIG_FILE=/app/sageworks_config.ini

# Install app specific dependencies
RUN pip install -r requirements.txt

# Run Gunicorn
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:8000", "--workers", "3"] & nginx -g 'daemon off;'
