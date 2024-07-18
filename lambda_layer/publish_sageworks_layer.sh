#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

BUCKET_NAME="sageworks-lambda-layer"
LAYER_NAME="sageworks_lambda_layer"
ZIP_FILE="sageworks_lambda_layer.zip"
S3_KEY="sageworks_lambda_layer.zip"

# Step 6: Create a ZIP Archive
echo "Creating ZIP archive..."

# Navigate to the output directory
cd sageworks_lambda_layer_output

# Remove existing ZIP file if it exists
if [ -f $ZIP_FILE ]; then
    echo "Removing existing ZIP file..."
    rm $ZIP_FILE
fi

# Create new ZIP file with quiet mode
zip -rq9 $ZIP_FILE python

# Step 7: Upload the ZIP file to S3
echo "Uploading ZIP file to S3..."
aws s3 cp $ZIP_FILE s3://$BUCKET_NAME/$S3_KEY

echo "Lambda layer ZIP file uploaded successfully to S3!"
