#!/bin/bash
set -e

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"

# Default image name
DEFAULT_IMAGE="aws_model_inference:0.1"
IMAGE_NAME=${1:-$DEFAULT_IMAGE}

# Port to use for testing
PORT=8080

echo "📋 Inference Container Test Script"
echo "======================================"

# Make sure test script exists
if [ ! -f "$SCRIPTS_DIR/test_inference.py" ]; then
  echo "❌ Error: test_inference.py not found in $SCRIPTS_DIR"
  exit 1
fi

# Start the inference container with proper log settings
echo "🚀 Starting inference container: $IMAGE_NAME"
CONTAINER_ID=$(docker run -d -p $PORT:$PORT -e PYTHONUNBUFFERED=1 "$IMAGE_NAME")

# Follow logs in the background
docker logs -f $CONTAINER_ID &
LOGS_PID=$!

# Ensure container and log process are stopped on script exit
function cleanup {
  echo "🧹 Stopping log process and container..."
  kill $LOGS_PID 2>/dev/null || true
  docker stop $CONTAINER_ID >/dev/null 2>&1
  docker rm $CONTAINER_ID >/dev/null 2>&1
}
trap cleanup EXIT

# Wait for container to initialize
echo "⏳ Waiting for server to initialize (5 seconds)..."
sleep 5

# Run the test
echo "🧪 Testing inference container..."
python "$SCRIPTS_DIR/test_inference.py" --host localhost --port $PORT

echo "======================================"