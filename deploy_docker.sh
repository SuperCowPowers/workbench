#!/bin/bash

# Deploy script for Docker image to ECR
# Usage: ./deploy.sh <version> [--stable]

set -e

# Check if version is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <version> [--stable]"
    echo "Example: $0 0_8_13 [--stable]"
    exit 1
fi

VERSION=$1
STABLE=false
ECR_REGISTRY="public.ecr.aws/m6i5k1r2"
IMAGE_NAME="workbench_base"
PLATFORM="linux/amd64"
CONFIG="open_source_config.json"

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

# Full version tag with architecture
VERSION_TAG="v${VERSION}_amd64"

echo "📦 Building Docker image version: $VERSION_TAG"
docker build --build-arg WORKBENCH_CONFIG=$CONFIG -t $IMAGE_NAME:$VERSION_TAG --platform $PLATFORM .

echo "🔑 Logging in to ECR..."
aws ecr-public get-login-password --region us-east-1 --profile $AWS_PROFILE | docker login --username AWS --password-stdin public.ecr.aws

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
