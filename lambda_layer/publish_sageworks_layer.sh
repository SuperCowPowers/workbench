#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

LAYER_NAME="sageworks-layer"

# Step 6: Create a ZIP Archive
echo "Creating ZIP archive..."
cd sageworks-lambda-layer-output
zip -r9 sageworks-lambda-layer.zip python

# Step 7: Publish the Lambda Layer
echo "Publishing Lambda layer..."
LAYER_VERSION=$(aws lambda publish-layer-version \
    --layer-name $LAYER_NAME \
    --description "SageWorks Lambda Layer" \
    --zip-file fileb://sageworks-lambda-layer.zip \
    --compatible-runtimes python3.10 \
    --query 'Version' \
    --output text)

# Step 9: Make the Lambda Layer Public
echo "Making Lambda layer public..."
aws lambda add-layer-version-permission \
    --layer-name $LAYER_NAME \
    --version-number $LAYER_VERSION \
    --statement-id public-access \
    --action lambda:GetLayerVersion \
    --principal '*'

# Output the ARN of the published Lambda layer
LAYER_ARN=$(aws lambda list-layer-versions \
    --layer-name $LAYER_NAME \
    --query 'LayerVersions[0].LayerVersionArn' \
    --output text)

echo "Lambda layer published successfully!"
echo "Layer ARN: $LAYER_ARN"
