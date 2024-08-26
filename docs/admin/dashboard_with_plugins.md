# Deploying Plugins with the Dashboard
Notes and information on how to include plugins with your SageWorks Dashboard.

- **ECR:** AWS Elastic Container Registry (stores Docker images)
- **ECS:** AWS Elastic Container Service (uses Docker images)

### Install Docker
  
- Linux: <https://docs.docker.com/engine/install/ubuntu/>
- Windows: <https://docs.docker.com/desktop/install/windows-install/>

## Build the Docker Image
If you don't already have a Dockerfile, here's one to get you started, just place this into your repo/directory that has the plugins. 

```
# Pull base sageworks dashboard image with specific tag (pick latest or stable)
FROM public.ecr.aws/m6i5k1r2/sageworks_dashboard:latest

# Copy the plugin files into the Dashboard plugins dir
COPY ./sageworks_plugins /app/sageworks_plugins
ENV SAGEWORKS_PLUGINS=/app/sageworks_plugins
```

**Note:** Your plugins directory should looks like this

```
sageworks_plugins/
   pages/
      my_plugin_page.py
      ...
   views/
      my_plugin_view.py
      ...
   web_components/
      my_component.py
      ...
```

### Build it

```
docker build -t my_sageworks_with_plugins:v1_0 --platform linux/amd64 .
```

### Test the Image Locally
You'll need to use AWS Credentials for this, it's a bit complicated, please contact SageWorks Support [sageworks@supercowpowers.com](mailto:sageworks@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 

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
docker tag my_sageworks_with_plugins:v1_0 \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/sageworks_with_plugins:v1_0
```
```
docker push \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/sageworks_with_plugins:v1_0
```

## Deploying Plugin Docker Image to AWS
Okay now that you have your plugin Docker Image you can deploy to your AWS account:

**Copy the Dashboard CDK files**

This is cheesy but just copy all the CDK files into your repo/directory.

```
cp -r sageworks/aws_setup/sageworks_dashboard_full /my/sageworks/stuff/
```

**Change the Docker Image to Deploy**

Now open up the `app.py` file and change this line to your Docker Image

```
# When you want a different docker image change this line
dashboard_image = "public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_8_3_amd64"
```

Make sure your `SAGEWORKS_CONFIG` is properly set, and run the following commands:

```
export SAGEWORKS_CONFIG=/Users/<user_name>/.sageworks/sageworks_config.json
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




### Note on SageWorks Configuration
All Configuration is managed by the CDK Python Script and the `SAGEWORKS_CONFIG` ENV var. If you want to change things like `REDIS_HOST` or `SAGEWORKS_BUCKET` you should do that with a `sageworks.config` file and then point the `SAGEWORKS_CONFIG` ENV var to that file.
