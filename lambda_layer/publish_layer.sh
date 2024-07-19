#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

BUCKET_NAME="sageworks-lambda-layer"
LAYER_NAME="sageworks_lambda_layer"
ZIP_FILE="sageworks_lambda_layer.zip"
S3_KEY="sageworks_lambda_layer.zip"

# Create a ZIP Archive for the SageWorks Lambda Layer
echo "Creating ZIP archive..."

# Navigate to the output directory
cd sageworks_lambda_layer_output

# Remove existing ZIP file if it exists
if [ -f $ZIP_FILE ]; then
    echo "Removing existing ZIP file..."
    rm $ZIP_FILE
fi

# Create new ZIP file with quiet mode
echo "Creating new ZIP file..."
zip -rq9 $ZIP_FILE python

# Navigate back to the original directory
cd ..

# Upload the ZIP file to S3
echo "Uploading ZIP file to S3..."
aws s3 cp sageworks_lambda_layer_output/$ZIP_FILE s3://$BUCKET_NAME/$S3_KEY

echo "Lambda layer ZIP file uploaded successfully to S3!"

# Publish the Lambda layer
layer_version=$(aws lambda publish-layer-version \
    --layer-name $LAYER_NAME \
    --description "SageWorks Lambda Layer" \
    --content S3Bucket=$BUCKET_NAME,S3Key=$S3_KEY \
    --compatible-runtimes python3.10 python3.11 python3.12 \
    --query 'Version' \
    --output text)

# Make the Lambda layer public
aws lambda add-layer-version-permission \
    --layer-name $LAYER_NAME \
    --version-number "$layer_version" \
    --statement-id public-access \
    --action lambda:GetLayerVersion \
    --principal "*"

# Query the ARN of the published Lambda layer
layer_arn=$(aws lambda list-layer-versions \
    --layer-name $LAYER_NAME \
    --query "LayerVersions[?Version==\`$layer_version\`].LayerVersionArn" \
    --output text)

echo "Lambda layer published successfully!"
echo "Layer ARN: $layer_arn"
