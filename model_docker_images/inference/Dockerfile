FROM python:3.12-slim

# Copy requirements file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your server code
COPY main.py /app/
WORKDIR /app

# Run the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]