# Deploying Plugins with the Dashboard
Notes and information on how to include plugins with your Workbench Dashboard.

- **ECR:** AWS Elastic Container Registry (stores Docker images)
- **ECS:** AWS Elastic Container Service (uses Docker images)

### Install Docker
  
- Linux: <https://docs.docker.com/engine/install/ubuntu/>
- Windows: <https://docs.docker.com/desktop/install/windows-install/>

## Build the Docker Image
If you don't already have a Dockerfile, here's one to get you started, just place this into your repo/directory that has the plugins. 

```
# Pull base workbench dashboard image with specific tag (pick latest or stable)
FROM public.ecr.aws/m6i5k1r2/workbench_dashboard:latest

# Copy the plugin files into the Dashboard plugins dir
COPY ./workbench_plugins /app/workbench_plugins
ENV WORKBENCH_PLUGINS=/app/workbench_plugins
```

**Note:** Your plugins directory should looks like this

```
workbench_plugins/
   pages/
      my_plugin_page.py
      ...
   views/
      my_plugin_view.py
      ...
   components/
      my_component.py
      ...
```

### Build it

```
docker build -t my_workbench_with_plugins:v1_0 --platform linux/amd64 .
```

### Test the Image Locally
You'll need to use AWS Credentials for this, it's a bit complicated, please contact Workbench Support [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

### Login to your ECR
Okay.. so after testing locally you're ready to push the Docker image (with Plugins) to the your ECR.

**Note:** This ECR should be **private** as your plugins are customized for specific business use cases.

Your ECR location will have this form
```
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
```

```
aws ecr get-login-password --region us-east-1 --profile <aws_profile> \
| docker login --username AWS --password-stdin \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
```

### Tag/Push the Image to AWS ECR
```
docker tag my_workbench_with_plugins:v1_0 \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/workbench_with_plugins:v1_0
```
```
docker push \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/workbench_with_plugins:v1_0
```

## Deploying Plugin Docker Image to AWS
Okay now that you have your plugin Docker Image you can deploy to your AWS account:

**Copy the Dashboard CDK files**

This is cheesy but just copy all the CDK files into your repo/directory.

```
cp -r workbench/aws_setup/workbench_dashboard_full /my/workbench/stuff/
```

**Change the Docker Image to Deploy**

Now open up the `app.py` file and change this line to your Docker Image

```
# When you want a different docker image change this line
dashboard_image = "public.ecr.aws/m6i5k1r2/workbench_dashboard:v0_8_3_amd64"
```

Make sure your `WORKBENCH_CONFIG` is properly set, and run the following commands:

```
export WORKBENCH_CONFIG=/Users/<user_name>/.workbench/workbench_config.json
cdk diff
cdk deploy
```

!!! warning "CDK Diff" 
    In particular, pay attention to the `cdk diff` it should **ONLY** have the image name as a difference.
    ```
    cdk diff
    [-] "Image": "<account>.dkr.ecr.us-east-1/my-plugins:latest_123",
    [+] "Image": "<account>.dkr.ecr.us-east-1/my-plugins:latest_456",
    ```




### Note on Workbench Configuration
All Configuration is managed by the CDK Python Script and the `WORKBENCH_CONFIG` ENV var. If you want to change things like `REDIS_HOST` or `WORKBENCH_BUCKET` you should do that with a `workbench.config` file and then point the `WORKBENCH_CONFIG` ENV var to that file.
