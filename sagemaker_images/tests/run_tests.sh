#!/usr/bin/env bash
set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
  echo "Usage: $(basename $0) IMAGE_TYPE [VERSION]"
  echo "  IMAGE_TYPE: base or pytorch_chem"
  echo "  VERSION:    Image version (default: 0.1)"
  echo ""
  echo "Examples:"
  echo "  $(basename $0) base"
  echo "  $(basename $0) base 0.2"
  echo "  $(basename $0) pytorch_chem 0.1"
  exit 1
}

if [ -z "$1" ]; then
  echo "Error: IMAGE_TYPE is required"
  usage
fi

IMAGE_TYPE=$1
IMAGE_VERSION=${2:-"0.1"}

case $IMAGE_TYPE in
  base)
    TRAINING_IMAGE="aws-ml-images/py312-base-training:${IMAGE_VERSION}"
    INFERENCE_IMAGE="aws-ml-images/py312-base-inference:${IMAGE_VERSION}"
    ;;
  pytorch_chem)
    TRAINING_IMAGE="aws-ml-images/py312-pytorch-chem-training:${IMAGE_VERSION}"
    INFERENCE_IMAGE="aws-ml-images/py312-pytorch-chem-inference:${IMAGE_VERSION}"
    ;;
  *)
    echo "Error: Unknown image type '$IMAGE_TYPE'"
    usage
    ;;
esac

# Test training container
echo "======================================"
echo -e "${YELLOW}Testing training container: ${TRAINING_IMAGE}${NC}"
echo "======================================"
python "$SCRIPT_DIR/test_training.py" --image "$TRAINING_IMAGE"

# Test inference container
echo "======================================"
echo -e "${YELLOW}Testing inference container: ${INFERENCE_IMAGE}${NC}"
echo "======================================"
python "$SCRIPT_DIR/test_inference.py" --image "$INFERENCE_IMAGE"

echo "======================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "======================================"
