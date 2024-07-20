# Use Python 3.10
FROM python:3.10.13

# Install SageWorks dependencies
COPY requirements.txt .
COPY requirements-no-dash.txt .
RUN pip install --no-cache-dir -r requirements-no-dash.txt

# Install latest Sageworks (no dependencies)
RUN pip install --no-cache-dir --no-deps 'sageworks[ml-tools,chem]'==0.7.0

