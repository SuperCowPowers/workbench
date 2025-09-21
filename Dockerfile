FROM python:3.12-slim

ARG WORKBENCH_VERSION=
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends libexpat1 libnghttp2-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir workbench${WORKBENCH_VERSION:+==$WORKBENCH_VERSION}
