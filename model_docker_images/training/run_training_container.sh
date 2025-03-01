#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PARENT_DIR/scripts"

# Make sure test_training.py exists
if [ ! -f "$SCRIPTS_DIR/test_training.py" ]; then
  echo "❌ Error: test_training.py not found in $SCRIPTS_DIR"
  exit 1
fi

IMAGE_NAME=${1:-aws_model_training:latest}

echo "🚀 Testing Training Container: $IMAGE_NAME"
python "$SCRIPTS_DIR/test_training.py" --image "$IMAGE_NAME"