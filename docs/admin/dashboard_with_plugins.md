# Including Plugins with the Dashboard
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
You'll need to use AWS Credentials for this, please contact SageWorks Support for help with this.

### Login to your ECR
Okay.. so after testing locally you're ready to push the Docker image (with Plugins) to the your ECR.

**Note:** This ECR should be **private** as your plugins are customized for specific business use cases.

Your ECR location will have this form
```
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
```

```
aws ecr-public get-login-password --region us-east-1 --profile \
<your_aws_profile> | docker login --username AWS \
--password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com
```
### Tag/Push the Image to AWS ECR
```
docker tag my_sageworks_with_plugins:v1_0 \
<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/sageworks_with_plugins:v1_0
```
```
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/sageworks_with_plugins:v1_0
```


### Note on SageWorks Configuration
Configuration is managed 'later' by the CDK Python Script, you should **not** set things like S3 bucket/API key/etc in the Docker image. When using CDK to deploy your SageWorks Dashboard it will manage the ECS Task definition so that the various configuration options are set.
