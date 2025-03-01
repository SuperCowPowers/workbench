#!/bin/bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# Get the project root directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
TRAINING_IMAGE="aws_model_training"
INFERENCE_IMAGE="aws_model_inference"
IMAGE_VERSION=${1:-"0.1"}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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