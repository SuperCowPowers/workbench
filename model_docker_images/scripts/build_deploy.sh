#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
TRAINING_DIR="$PROJECT_ROOT/training"
INFERENCE_DIR="$PROJECT_ROOT/inference"
TRAINING_IMAGE="aws_model_training"
INFERENCE_IMAGE="aws_model_inference"
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

# Function to build a Docker image
build_image() {
    local dir=$1
    local image_name=$2
    local tag=$3
    local full_name="${image_name}:${tag}"

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
    local image_name=$1
    local tag=$2
    local use_latest=$3
    local full_name="${image_name}:${tag}"

    for REGION in "${REGION_LIST[@]}"; do
        echo "Processing region: ${REGION}"
        # Construct the ECR repository URL (using your account ID 507740646243)
        ECR_REPO="507740646243.dkr.ecr.${REGION}.amazonaws.com/model_images/${image_name}"
        AWS_ECR_IMAGE="${ECR_REPO}:${tag}"

        echo "Logging in to AWS ECR in ${REGION}..."
        aws ecr get-login-password --region ${REGION} --profile ${AWS_PROFILE} | \
            docker login --username AWS --password-stdin ${ECR_REPO}

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

# Build training image
echo "======================================"
echo "🏗️  Building training container"
echo "======================================"
build_image "$TRAINING_DIR" "$TRAINING_IMAGE" "$IMAGE_VERSION"

# Build inference image
echo "======================================"
echo "🏗️  Building inference container"
echo "======================================"
build_image "$INFERENCE_DIR" "$INFERENCE_IMAGE" "$IMAGE_VERSION"

echo "======================================"
echo -e "${GREEN}✅ All builds completed successfully!${NC}"
echo "======================================"

if [ "$DEPLOY" = true ]; then
    echo "======================================"
    echo "🚀 Deploying containers to ECR"
    echo "======================================"

    # Deploy training image
    echo "Deploying training image..."
    deploy_image "$TRAINING_IMAGE" "$IMAGE_VERSION" "$LATEST"

    # Deploy inference image
    echo "Deploying inference image..."
    deploy_image "$INFERENCE_IMAGE" "$IMAGE_VERSION" "$LATEST"

    echo "======================================"
    echo -e "${GREEN}✅ Deployment complete!${NC}"
    echo "======================================"
else
    echo "Local build complete. Use --deploy to push the images to AWS ECR in regions: ${REGION_LIST[*]}."

    # Print information about the built images
    echo "======================================"
    echo "📋 Image information:"
    echo "Training image: ${TRAINING_IMAGE}:${IMAGE_VERSION}"
    echo "Inference image: ${INFERENCE_IMAGE}:${IMAGE_VERSION}"
    echo "======================================"

    # Ask if user wants to test the containers
    read -p "Do you want to test the containers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Test training container
        echo "======================================"
        echo "🧪 Testing training container"
        echo "======================================"
        python "$SCRIPT_DIR/test_training.py" --image "${TRAINING_IMAGE}:${IMAGE_VERSION}"

        # Test inference container
        echo "======================================"
        echo "🧪 Testing inference container"
        echo "======================================"

        # Start the inference container in the background
        echo "Starting inference container..."
        CONTAINER_ID=$(docker run -d -p 8080:8080 "${INFERENCE_IMAGE}:${IMAGE_VERSION}")

        # Wait for the container to initialize
        echo "Waiting for server to initialize (5 seconds)..."
        sleep 5

        # Run the test
        python "$SCRIPT_DIR/test_inference.py"

        # Stop and remove the container
        echo "Stopping inference container..."
        docker stop $CONTAINER_ID
        docker rm $CONTAINER_ID

        echo "======================================"
        echo -e "${GREEN}✅ Testing completed!${NC}"
        echo "======================================"
    fi
fi