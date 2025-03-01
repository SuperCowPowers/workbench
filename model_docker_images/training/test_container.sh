#!/bin/bash
set -e

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Default image name with latest tag
DEFAULT_IMAGE="aws_model_training:0.1"
IMAGE_NAME=${1:-$DEFAULT_IMAGE}

echo "📋 Training Container Test Script"
echo "======================================"

# Make sure test_training.py exists
if [ ! -f "$SCRIPTS_DIR/test_training.py" ]; then
  echo "❌ Error: test_training.py not found in $SCRIPTS_DIR"
  exit 1
fi

echo "🚀 Testing Training Container: $IMAGE_NAME"
python "$SCRIPTS_DIR/test_training.py" --image "$IMAGE_NAME"

echo "======================================"