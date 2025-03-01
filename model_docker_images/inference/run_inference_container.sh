#!/bin/bash
set -e

echo "🚀 Starting AWS Model Inference Container..."
docker run -d -p 8080:8080 --name aws_model_test aws_model_image:0.1

echo "⏳ Waiting for server to initialize (5 seconds)..."
sleep 5

echo "🧪 Running tests against the server..."
python test_inference.py

echo "🧹 Cleaning up - stopping and removing container..."
docker stop aws_model_test
docker rm aws_model_test

echo "✅ Done!"