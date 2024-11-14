# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm

# Update the package list, upgrade packages, and handle vulnerabilities in one step
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libexpat1 libnghttp2-dev && \
    apt-get remove --purge -y git libaom3 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements and install dependencies (except SageWorks)
COPY requirements-no-dash.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-no-dash.txt

# Install SageWorks on its own layer
RUN pip install --no-cache-dir 'sageworks[ml-tool,chem]'==0.8.58
