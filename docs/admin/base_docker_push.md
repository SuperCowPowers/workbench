# Workbench Base Docker Build and Push

Notes and information on how to do the Docker Builds and Push to AWS ECR.

### Update Workbench Version
```
vi Dockerfile

# Install latest Workbench
RUN pip install --no-cache-dir 'workbench[ml-tool,chem]'==0.7.0
```

### Build the Docker Image
**Note:** For a client specific config file you'll need to copy it locally so that it's within Dockers 'build context'. If you're building the 'vanilla' open source Docker image, then you can use the `open_source_config.json` that's in the directory already.

```
docker build --build-arg WORKBENCH_CONFIG=open_source_config.json -t \
workbench_base:v0_7_0_amd64 --platform linux/amd64 .
```

### Test the Image Locally
You have a `docker_local_base` alias in your `~/.zshrc` :)

### Login to ECR
```
aws ecr-public get-login-password --region us-east-1 --profile \
scp_sandbox_admin | docker login --username AWS \
--password-stdin public.ecr.aws
```
### Tag/Push the Image to AWS ECR
```
docker tag workbench_base:v0_7_0_amd64 \
public.ecr.aws/m6i5k1r2/workbench_base:v0_7_0_amd64
```
```
docker push public.ecr.aws/m6i5k1r2/workbench_base:v0_7_0_amd64
```

### Update the 'latest' tag
```
docker tag public.ecr.aws/m6i5k1r2/workbench_base:v0_7_0_amd64 \
public.ecr.aws/m6i5k1r2/workbench_base:latest
```
```
docker push public.ecr.aws/m6i5k1r2/workbench_base:latest
```

### Update the 'stable' tag
This is obviously only when you want to mark a version as stable. Meaning that it seems to 'be good and stable (ish)' :)

```
docker tag public.ecr.aws/m6i5k1r2/workbench_base:v0_7_0_amd64 \
public.ecr.aws/m6i5k1r2/workbench_base:stable
```
```
docker push public.ecr.aws/m6i5k1r2/workbench_base:stable
```

### Test the ECR Image
You have a `docker_ecr_base` alias in your `~/.zshrc` :)


