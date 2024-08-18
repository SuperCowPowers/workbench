# Use Python 3.12 as the base image
FROM python:3.12.5-bookworm

# Remove git from the base image (vulnerability)
RUN apt-get remove --purge -y git && apt-get autoremove -y && apt-get clean

# Remove AOM from the base image (vulnerability)
RUN apt-get remove --purge -y libaom3 && apt-get autoremove -y && apt-get clean

# Upgrade the nghttp2 package to fix a vulnerability
RUN apt-get update && apt-get install -y libnghttp2-dev && apt-get clean

# Install SageWorks dependencies
COPY requirements.txt .
COPY requirements-no-dash.txt .
RUN pip install --no-cache-dir -r requirements-no-dash.txt

# Install latest Sageworks (no dependencies)
RUN pip install --no-cache-dir --no-deps 'sageworks[ml-tools,chem]'==0.8.2

# Remove pip (vulnerability)
RUN python -m pip uninstall -y pip && \
    rm -rf /usr/local/lib/python*/dist-packages/pip /usr/local/bin/pip* && \
    apt-get autoremove -y && apt-get clean \
