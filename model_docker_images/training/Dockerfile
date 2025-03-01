FROM python:3.12-slim

# Copy requirements file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set up the program in the image
COPY train.py /opt/program/
WORKDIR /opt/program

# Set up the entry point
ENTRYPOINT ["python", "train.py"]
