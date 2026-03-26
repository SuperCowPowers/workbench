#!/usr/bin/env bash
set -e

# Get the directory of this script (sagemaker_images/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# AWS Account ID
AWS_ACCOUNT_ID="507740646243"

# Map of image types to their repository names
# Image types use nested paths (e.g., base/training) matching directory structure
declare -A REPO_MAP=(
  ["base/training"]="aws-ml-images/py312-base-training"
  ["base/inference"]="aws-ml-images/py312-base-inference"
  ["pytorch_chem/training"]="aws-ml-images/py312-pytorch-chem-training"
  ["pytorch_chem/inference"]="aws-ml-images/py312-pytorch-chem-inference"
  ["ml_pipelines"]="aws-ml-images/py312-ml-pipelines"
)

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
usage() {
  echo "Usage: $(basename $0) IMAGE_TYPE [VERSION] [--deploy] [--overwrite]"
  echo "  IMAGE_TYPE: One of ${!REPO_MAP[*]}"
  echo "  VERSION: Image version (default: 0.1)"
  echo "  --deploy: Deploy to ECR (also updates 'latest' tag)"
  echo "  --overwrite: Overwrite existing versioned images"
  exit 1
}

# Validate image type
if [ -z "$1" ] || [[ ! -v "REPO_MAP[$1]" ]]; then
  echo "Error: You must specify a valid image type"
  usage
fi

IMAGE_TYPE=$1
shift

# Set defaults
IMAGE_VERSION="0.1"
DEPLOY=false
OVERWRITE=false

# Parse remaining arguments
[[ $1 =~ ^[0-9]+\.[0-9]+$ ]] && IMAGE_VERSION=$1 && shift

for arg in "$@"; do
  case $arg in
    --deploy)    DEPLOY=true ;;
    --overwrite) OVERWRITE=true ;;
    *)           echo "Unknown option: $arg" && usage ;;
  esac
done

# Check AWS_PROFILE when deploying
if [ "$DEPLOY" = true ]; then
  : "${AWS_PROFILE:?AWS_PROFILE environment variable is not set.}"
fi

# Define the regions to deploy to
REGION_LIST=("us-east-1" "us-west-2")

# Get repository and directory for the selected image type
REPO_NAME=${REPO_MAP[$IMAGE_TYPE]}
DIR=$SCRIPT_DIR/$IMAGE_TYPE

# Function to build a Docker image
build_image() {
  local arch=$1  # amd64 or arm64
  local tag=$2
  local platform="linux/$arch"
  local name="$REPO_NAME:$tag"

  echo -e "${YELLOW}Building image: $name ($platform)${NC}"

  if [ ! -f "$DIR/Dockerfile" ]; then
    echo "❌  Error: Dockerfile not found in $DIR"
    exit 1
  fi

  # Copy pyproject.toml into build context for dependency layer caching (if needed by Dockerfile)
  if grep -q 'pyproject.toml' "$DIR/Dockerfile"; then
    cp "$SCRIPT_DIR/../pyproject.toml" "$SCRIPT_DIR/pyproject.toml"
    trap 'rm -f "$SCRIPT_DIR/pyproject.toml"' EXIT
  fi

  # Build with sagemaker_images/ as context, using -f for the nested Dockerfile
  docker build --platform $platform -t $name -f $DIR/Dockerfile $SCRIPT_DIR
  echo -e "${GREEN}✅  Successfully built: $name${NC}"
}

# Helper function to check if an image exists in ECR
image_exists() {
  local repo=$1
  local tag=$2
  local region=$3

  aws ecr describe-images \
    --repository-name $repo \
    --image-ids imageTag=$tag \
    --region $region \
    --profile $AWS_PROFILE &>/dev/null

  return $?
}

# Function to deploy an image to ECR
deploy_image() {
  local tag=$1
  local full_name="$REPO_NAME:$tag"

  for region in "${REGION_LIST[@]}"; do
    echo "Processing region: $region"

    # Construct ECR repository URL
    local ecr_repo="$AWS_ACCOUNT_ID.dkr.ecr.$region.amazonaws.com/$REPO_NAME"
    local ecr_image="$ecr_repo:$tag"

    # Login to ECR
    echo "Logging in to AWS ECR in $region..."
    aws ecr get-login-password --region $region --profile $AWS_PROFILE | \
      docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$region.amazonaws.com"

    # Check if versioned image exists (skip if exists and no overwrite)
    if [ "$OVERWRITE" = false ] && image_exists $REPO_NAME $tag $region; then
      echo "Image $ecr_image already exists and overwrite is set to false. Skipping..."
      continue
    fi

    # Tag and push versioned image
    echo "Tagging image for AWS ECR as $ecr_image..."
    docker tag $full_name $ecr_image

    echo "Pushing Docker image to AWS ECR: $ecr_image..."
    docker push $ecr_image

    # Always update the latest tag
    local ecr_latest="$ecr_repo:latest"
    echo "Tagging AWS ECR image as latest: $ecr_latest..."
    docker tag $full_name $ecr_latest

    echo "Pushing Docker image to AWS ECR: $ecr_latest..."
    docker push $ecr_latest
  done
}

# Build AMD64 image
echo "======================================"
echo "🏗️  Building $IMAGE_TYPE container (AMD64)"
echo "======================================"
build_image "amd64" "$IMAGE_VERSION"

echo "======================================"
echo -e "${GREEN}✅  Build completed successfully!${NC}"
echo "======================================"

# Deploy if requested
if [ "$DEPLOY" = true ]; then
  echo "======================================"
  echo "🚀  Deploying $IMAGE_TYPE container to ECR"
  echo "======================================"

  deploy_image "$IMAGE_VERSION"

  echo "======================================"
  echo -e "${GREEN}✅  Deployment complete!${NC}"
  echo "======================================"
else
  # Print information about the built image
  echo "Local build complete. Use --deploy to push the image to AWS ECR in regions: ${REGION_LIST[*]}."

  echo "======================================"
  echo "📋  Image information:"
  echo "$IMAGE_TYPE image: $REPO_NAME:$IMAGE_VERSION"
  echo "======================================"
  echo "To test these containers, run: $SCRIPT_DIR/tests/run_tests.sh $IMAGE_VERSION"
fi
