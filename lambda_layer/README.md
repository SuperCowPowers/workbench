# Building a Lambda Layer for SageWorks

### Step 1: Build the Docker Container

```bash
docker build -t sageworks-lambda-layer .
```

### Step 2: Run the Docker Container with an Interactive Shell

```bash
docker run --platform linux/amd64 -it sageworks-lambda-layer /bin/bash
```

### Step 3: Verify the Installation

Inside the container, check the contents of the `/asset-output/python` directory:

```bash
ls -l /asset-output/python
```

Ensure the `sageworks` package and its dependencies are installed correctly.

### Step 4: Exit the Container

Exit the interactive shell:

```bash
exit
```

### Step 5: Copy the Results to Your Local Disk

First, find the container ID:

```bash
docker ps -a
```

Locate the container ID of the `sageworks-lambda-layer` container, then copy the contents:

```bash
docker cp <container_id>:/asset-output ./sageworks_lambda_layer_output/
```

## Publish the Lambda Layer

Run the `publish_lambda_layer.sh` script to create a ZIP archive, publish the Lambda layer, and make it public:

```bash
./publish_lambda_layer.sh
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
