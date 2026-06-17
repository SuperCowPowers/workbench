#!/bin/bash

# Deploy script for Docker image to ECR
# Usage: ./deploy.sh <version> [--stable] [--overwrite]

set -e

# Check if version is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <version> [--stable] [--overwrite]"
    echo "Example: $0 0_8_13 [--stable] [--overwrite]"
    exit 1
fi

VERSION=$1
STABLE=false
OVERWRITE=false
ECR_REGISTRY="public.ecr.aws/m6i5k1r2"
IMAGE_NAME="workbench_dashboard"
PLATFORM="linux/amd64"
CONFIG="open_source_config.json"

# Resolve script dir so cache/builder setup is independent of the caller's cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Local BuildKit cache. ECR Public can't host a registry build cache, so we
# persist BuildKit's layer cache to a gitignored local dir instead. A cold
# build restores the heavy locked-deps layer from disk instead of re-resolving
# every wheel. mode=max captures intermediate stages, not just the final image.
CACHE_DIR="$SCRIPT_DIR/.buildcache"

# A docker-container builder is required: the default `docker` driver can't
# export a local cache (--cache-to). Create one if missing and target it
# explicitly with --builder so we don't disturb the user's active builder.
BUILDER="workbench-dashboard-builder"
if ! docker buildx inspect "$BUILDER" &> /dev/null; then
    echo "🔧 Creating buildx builder: $BUILDER"
    docker buildx create --name "$BUILDER" --driver docker-container
fi

# Get AWS_PROFILE from environment
if [ -z "$AWS_PROFILE" ]; then
    echo "Error: AWS_PROFILE environment variable is not set"
    exit 1
fi

# Check if stable flag is provided
if [[ "$*" == *--stable* ]]; then
    STABLE=true
    echo "This build will be tagged as stable"
fi

# Check if overwrite flag is provided
if [[ "$*" == *--overwrite* ]]; then
    OVERWRITE=true
fi

# Full version tag with architecture
VERSION_TAG="v${VERSION}_amd64"

echo "📦 Building Docker image version: $VERSION_TAG"

docker buildx build \
    --builder $BUILDER \
    --platform $PLATFORM \
    --build-arg WORKBENCH_CONFIG=$CONFIG \
    --cache-from type=local,src=$CACHE_DIR \
    --cache-to type=local,dest=$CACHE_DIR,mode=max \
    -t $IMAGE_NAME:$VERSION_TAG \
    --load \
    .

echo "🔑 Logging in to ECR..."
aws ecr-public get-login-password --region us-east-1 --profile $AWS_PROFILE | docker login --username AWS --password-stdin public.ecr.aws

# Guard against silently overwriting an existing version. We abort before any
# push so the moving 'latest'/'stable' tags don't make it look like a real
# deploy happened. Pass --overwrite to intentionally replace. (ECR Public lives
# only in us-east-1.)
if [ "$OVERWRITE" = false ] && \
   aws ecr-public describe-images \
     --repository-name $IMAGE_NAME \
     --image-ids imageTag=$VERSION_TAG \
     --region us-east-1 --profile $AWS_PROFILE &> /dev/null; then
    echo "❌ Version $VERSION_TAG already exists in ECR. Bump the version or pass --overwrite to replace it."
    exit 1
fi

# Set full image path with tag
FULL_IMAGE_PATH="$ECR_REGISTRY/$IMAGE_NAME:$VERSION_TAG"

echo "🏷️ Tagging image with version: $VERSION_TAG"
docker tag $IMAGE_NAME:$VERSION_TAG $FULL_IMAGE_PATH

echo "⬆️ Pushing image with version tag: $VERSION_TAG"
docker push $FULL_IMAGE_PATH

echo "🏷️ Tagging and pushing as 'latest'"
docker tag $FULL_IMAGE_PATH $ECR_REGISTRY/$IMAGE_NAME:latest
docker push $ECR_REGISTRY/$IMAGE_NAME:latest

# If stable flag is provided, tag and push as stable
if [ "$STABLE" = true ]; then
    echo "🌟 Tagging and pushing as 'stable'"
    docker tag $FULL_IMAGE_PATH $ECR_REGISTRY/$IMAGE_NAME:stable
    docker push $ECR_REGISTRY/$IMAGE_NAME:stable
fi

echo "✅ Deployment complete!"
echo "   - Version: $VERSION_TAG"
echo "   - Latest: yes"
echo "   - Stable: $STABLE"
