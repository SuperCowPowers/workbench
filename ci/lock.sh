#!/usr/bin/env bash
# Regenerate per-image requirements.lock files from pyproject.toml (+ optional
# per-image requirements.in for image-only deps like fastapi/uvicorn).
#
# Run after bumping deps in pyproject.toml or changing image dep selections.
# CI verifies the lockfiles are in sync by running this and checking
# `git diff --exit-code`.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_VERSION=3.12
PYTHON_PLATFORM=x86_64-unknown-linux-gnu
PYTORCH_CPU=https://download.pytorch.org/whl/cpu
PYTORCH_CUDA=https://download.pytorch.org/whl/cu130

# By default uv pip compile reuses the versions already pinned in each output
# lock as preferred-version hints, so re-running only changes what a constraint
# edit in pyproject.toml / requirements.in actually forces. New upstream
# releases don't cause churn.
#
# Escape hatches when you do want movement:
#   UPGRADE=1 ./ci/lock.sh           re-resolve everything to the latest (the
#                                    old aggressive behavior; --no-cache forces
#                                    fresh PyPI metadata, --upgrade drops the
#                                    preferred-versions hint)
#   add --upgrade-package <name>     bump a single package on a compile call
if [[ "${UPGRADE:-}" == "1" ]]; then
    FRESH="--no-cache --upgrade"
else
    FRESH=""
fi

# Compile one image's lockfile.
#   $1: output path
#   $@: additional uv pip compile args (extras, extra-index-url, requirements.in inputs, etc.)
compile() {
    local out="$1"
    shift
    echo "==> $out"
    uv pip compile pyproject.toml \
        --python-version "$PYTHON_VERSION" \
        --python-platform "$PYTHON_PLATFORM" \
        $FRESH \
        -o "$out" \
        "$@"
}

# ml_pipelines: full workbench + [misc], CPU torch (pulled in transitively via sagemaker-serve).
# No requirements.in needed — pyproject + [misc] covers everything; torch arrives transitively.
compile sagemaker_images/ml_pipelines/requirements.lock \
    --extra misc \
    --extra-index-url "$PYTORCH_CPU" \
    --index-strategy unsafe-best-match

# aws_dashboard: full workbench + [misc] + [ui] + uvicorn/asgiref overlay, CPU torch
compile applications/aws_dashboard/requirements.lock \
    applications/aws_dashboard/requirements.in \
    --extra misc --extra ui \
    --extra-index-url "$PYTORCH_CPU" \
    --index-strategy unsafe-best-match

# base/inference: endpoint runtime — strict subset of workbench main deps
# (no sagemaker SDK, no orchestration). Lockfile generated from
# requirements.in alone, NOT from pyproject (would over-include).
# Override the default `uv pip compile pyproject.toml ...` invocation.
echo "==> sagemaker_images/base/inference/requirements.lock"
uv pip compile sagemaker_images/base/inference/requirements.in \
    --python-version "$PYTHON_VERSION" \
    --python-platform "$PYTHON_PLATFORM" \
    $FRESH \
    -o sagemaker_images/base/inference/requirements.lock

# base/training: same strict-subset pattern as base/inference, minus fastapi/uvicorn
# (training jobs don't serve a server).
echo "==> sagemaker_images/base/training/requirements.lock"
uv pip compile sagemaker_images/base/training/requirements.in \
    --python-version "$PYTHON_VERSION" \
    --python-platform "$PYTHON_PLATFORM" \
    $FRESH \
    -o sagemaker_images/base/training/requirements.lock

# pytorch_chem/inference: smoke-contract subset + chemprop + fastapi/uvicorn,
# CPU torch. Chemprop inference on small batches (1–100 molecules per request,
# typical real-time scoring) is dominated by CPU-friendly work; GPU adds
# launch/transfer overhead without meaningful speedup. Keeps the image ~3GB
# smaller and avoids the slim-base + CUDA-wheel mismatch. Switch to
# $PYTORCH_CUDA + nvidia/cuda base if you ever add a high-throughput
# virtual-screening endpoint.
echo "==> sagemaker_images/pytorch_chem/inference/requirements.lock"
uv pip compile sagemaker_images/pytorch_chem/inference/requirements.in \
    --python-version "$PYTHON_VERSION" \
    --python-platform "$PYTHON_PLATFORM" \
    --extra-index-url "$PYTORCH_CPU" \
    --index-strategy unsafe-best-match \
    $FRESH \
    -o sagemaker_images/pytorch_chem/inference/requirements.lock

# pytorch_chem/training: smoke-contract subset + chemprop + shap, CUDA torch,
# no fastapi/uvicorn (training doesn't serve).
echo "==> sagemaker_images/pytorch_chem/training/requirements.lock"
uv pip compile sagemaker_images/pytorch_chem/training/requirements.in \
    --python-version "$PYTHON_VERSION" \
    --python-platform "$PYTHON_PLATFORM" \
    --extra-index-url "$PYTORCH_CUDA" \
    --index-strategy unsafe-best-match \
    $FRESH \
    -o sagemaker_images/pytorch_chem/training/requirements.lock

# compound_explorer: full workbench + [misc] + [ui] + shap + uvicorn/asgiref overlay, CPU torch
# Mirrors aws_dashboard (same uvicorn+asgiref serving stack) plus shap for the explorer's SHAP plots.
compile applications/compound_explorer/requirements.lock \
    applications/compound_explorer/requirements.in \
    --extra misc --extra ui \
    --extra-index-url "$PYTORCH_CPU" \
    --index-strategy unsafe-best-match

echo
echo "Done. Verify with: git diff sagemaker_images/*/requirements.lock applications/*/requirements.lock"
