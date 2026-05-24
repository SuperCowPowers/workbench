# Multi-stage build for dependency layer isolation. Layer 1 has all the heavy
# pip deps (auto-extracted from workbench's pyproject.toml so we don't have to
# maintain a duplicated list). Layer 2 installs workbench itself with --no-deps
# so per-version-bump deltas are small (~20MB instead of full pip resolve).
FROM python:3.12-slim AS builder

# Build deps for any packages that need compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc

# Layer 1: Workbench's transitive deps + [modeling] extras.
# Auto-extract from pyproject.toml so we don't maintain a duplicated dep list.
COPY pyproject.toml /tmp/pyproject.toml
RUN python -c "\
import tomllib; \
data = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
deps = data['project']['dependencies'] + data['project']['optional-dependencies']['modeling']; \
open('/tmp/workbench_deps.txt', 'w').write('\n'.join(deps))" && \
    pip install --no-cache-dir -r /tmp/workbench_deps.txt && \
    rm /tmp/pyproject.toml /tmp/workbench_deps.txt

# Final runtime image
FROM python:3.12-slim

# Runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy heavy deps from builder (changes rarely)
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Layer 2: Workbench + bridges — rebuilds per version bump, ~20MB delta.
# workbench-bridges==0.2.10 is retained for backward compatibility with existing
# deployed-endpoint model bundles; drop in a future image rev once no production
# endpoint references workbench_bridges.
ARG WORKBENCH_VERSION=0.8.345
RUN pip install --no-cache-dir --no-deps "workbench==${WORKBENCH_VERSION}" && \
    pip install --no-cache-dir --no-deps "workbench-bridges==0.2.10"
