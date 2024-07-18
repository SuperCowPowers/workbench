# Building a Lambda Layer for SageWorks

### Step 1: Build the Docker Container

```bash
docker build --platform linux/amd64 -t sageworks-lambda-layer .
```

### Step 2: Run the Docker Container with an Interactive Shell

```bash
docker run --name sageworks-layer-container -it sageworks-lambda-layer
```

### Step 3: Look at the size of the Python Packages Installation

```
cd python
du -sh * | sort -h
```

### Step 4: Exit the Container

```bash
exit
```

### Step 5: Copy the Results to Your Local Disk

```bash
docker cp sageworks-layer-container:/asset-output ./sageworks_lambda_layer_output/
```

## Publish the Lambda Layer

Run the `publish_lambda_layer.sh` script to create a ZIP archive, publish the Lambda layer, and make it public:

```bash
./publish_sageworks_layer.sh
```

The script will output the ARN of the published Lambda layer.

### Using the Lambda Layer in Other AWS Accounts

To use the Lambda layer in other AWS accounts, include the layer ARN in your Lambda function configuration:

```bash
aws lambda update-function-configuration \
    --function-name your-lambda-function-name \
    --layers <layer-arn>
```

Replace `<layer-arn>` with the ARN provided by the script.
