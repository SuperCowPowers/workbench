#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# AWS Account ID
AWS_ACCOUNT_ID="507740646243"

# Define repository names - used for both local and ECR images
TRAINING_REPO="aws-ml-images/py312-sklearn-xgb-training"
INFERENCE_REPO="aws-ml-images/py312-sklearn-xgb-inference"

# Local directories
TRAINING_DIR="$PROJECT_ROOT/training"
INFERENCE_DIR="$PROJECT_ROOT/inference"

# Image version
IMAGE_VERSION=${1:-"0.1"}

# Expect AWS_PROFILE to be set in the environment when deploying
if [ "$2" == "--deploy" ]; then
    : "${AWS_PROFILE:?AWS_PROFILE environment variable is not set.}"
fi

# Define the regions to deploy to.
REGION_LIST=("us-east-1" "us-west-2")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
DEPLOY=false
LATEST=false
for arg in "$@"; do
    case $arg in
        --deploy)
            DEPLOY=true
            ;;
        --latest)
            LATEST=true
            ;;
        *)
            ;;
    esac
done

# Function to build a Docker image (AMD64)
build_image() {
    local dir=$1
    local repo_name=$2
    local tag=$3
    local full_name="${repo_name}:${tag}"

    echo -e "${YELLOW}Building image: ${full_name}${NC}"

    # Check if Dockerfile exists
    if [ ! -f "$dir/Dockerfile" ]; then
        echo "❌ Error: Dockerfile not found in $dir"
        return 1
    fi

    # Build the image for AMD64 architecture
    echo "Building local Docker image ${full_name} for linux/amd64..."
    docker build --platform linux/amd64 -t $full_name $dir

    echo -e "${GREEN}✅ Successfully built: ${full_name}${NC}"
    return 0
}

# Function to deploy an image to ECR
deploy_image() {
    local repo_name=$1
    local tag=$2
    local use_latest=$3
    local full_name="${repo_name}:${tag}"

    for REGION in "${REGION_LIST[@]}"; do
        echo "Processing region: ${REGION}"
        # Construct the ECR repository URL
        ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${repo_name}"
        AWS_ECR_IMAGE="${ECR_REPO}:${tag}"

        echo "Logging in to AWS ECR in ${REGION}..."
        aws ecr get-login-password --region ${REGION} --profile ${AWS_PROFILE} | \
            docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

        echo "Tagging image for AWS ECR as ${AWS_ECR_IMAGE}..."
        docker tag ${full_name} ${AWS_ECR_IMAGE}

        echo "Pushing Docker image to AWS ECR: ${AWS_ECR_IMAGE}..."
        docker push ${AWS_ECR_IMAGE}

        if [ "$use_latest" = true ]; then
            AWS_ECR_LATEST="${ECR_REPO}:latest"
            echo "Tagging AWS ECR image as latest: ${AWS_ECR_LATEST}..."
            docker tag ${full_name} ${AWS_ECR_LATEST}
            echo "Pushing Docker image to AWS ECR: ${AWS_ECR_LATEST}..."
            docker push ${AWS_ECR_LATEST}
        fi
    done
}

# Build training image (AMD64)
echo "======================================"
echo "🏗️  Building training container"
echo "======================================"
build_image "$TRAINING_DIR" "$TRAINING_REPO" "$IMAGE_VERSION"

# Build inference image (AMD64)
echo "======================================"
echo "🏗️  Building inference container (AMD64)"
echo "======================================"
build_image "$INFERENCE_DIR" "$INFERENCE_REPO" "$IMAGE_VERSION"

# Build inference image for ARM64 ---
echo "======================================"
echo "🏗️  Building inference container (ARM64)"
echo "======================================"
if [ ! -f "$INFERENCE_DIR/Dockerfile" ]; then
    echo "❌ Error: Dockerfile not found in $INFERENCE_DIR"
    exit 1
fi
echo "Building local Docker image ${INFERENCE_REPO}:${IMAGE_VERSION}-arm64 for linux/arm64..."
docker build --platform linux/arm64 -t ${INFERENCE_REPO}:${IMAGE_VERSION}-arm64 $INFERENCE_DIR
echo -e "${GREEN}✅ Successfully built: ${INFERENCE_REPO}:${IMAGE_VERSION}-arm64${NC}"

echo "======================================"
echo -e "${GREEN}✅ All builds completed successfully!${NC}"
echo "======================================"

if [ "$DEPLOY" = true ]; then
    echo "======================================"
    echo "🚀 Deploying containers to ECR"
    echo "======================================"

    # Deploy training image
    echo "Deploying training image..."
    deploy_image "$TRAINING_REPO" "$IMAGE_VERSION" "$LATEST"

    # Deploy inference images
    echo "Deploying inference image (AMD64)..."
    deploy_image "$INFERENCE_REPO" "$IMAGE_VERSION" "$LATEST"

    echo "Deploying inference image (ARM64)..."
    deploy_image "$INFERENCE_REPO" "${IMAGE_VERSION}-arm64" "$LATEST"

    echo "======================================"
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    echo "======================================"
else
    echo "Local build complete. Use --deploy to push the images to AWS ECR in regions: ${REGION_LIST[*]}."

    # Print information about the built images
    echo "======================================"
    echo "📋 Image information:"
    echo "Training image: ${TRAINING_REPO}:${IMAGE_VERSION}"
    echo "Inference image (AMD64): ${INFERENCE_REPO}:${IMAGE_VERSION}"
    echo "Inference image (ARM64): ${INFERENCE_REPO}:${IMAGE_VERSION}-arm64"
    echo "======================================"

    # Inform about testing option
    echo "To test these containers, run: $PROJECT_ROOT/tests/run_tests.sh ${IMAGE_VERSION}"
fi
