# Docker Build and Push

Notes and information on how to do the Docker Builds and Push to AWS ECR.

The following instructions should work, but things change :)

### Update SageWorks Version
```
cd applications/aws_dashboard
vi Dockerfile

# Install Sageworks (changes often)
RUN pip install --no-cache-dir sageworks==0.4.13 <-- change this
```

### Build the Docker Image
**Note:** For a client specific config file you'll need to copy it locally so that it's within Dockers 'build context'. If you're building the 'vanilla' open source Docker image, then you can use the `open_source_config.json` that's in the directory already.

```
docker build --build-arg SAGEWORKS_CONFIG=open_source_config.json -t \
sageworks_dashboard:v0_4_13_amd64 --platform linux/amd64 .
```

**Docker with Custom Plugins:** If you're using custom plugins you may want to change the SAGEWORKS_PLUGINS directory to something like `/app/sageworks_plugins` and then have Dockerfile copy your plugins into that directory on the Docker image.

### Test the Image Locally
You have a `docker_local_dashboard` alias in your `~/.zshrc` :)

### Login to ECR
```
aws ecr-public get-login-password --region us-east-1 --profile \
scp_sandbox_admin | docker login --username AWS \
--password-stdin public.ecr.aws
```
### Tag/Push the Image to AWS ECR
```
docker tag sageworks_dashboard:v0_4_13_amd64 \
public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_4_13_amd64
```
```
docker push public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_4_13_amd64
```

### Update the 'latest' tag
```
docker tag public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_4_13_amd64 \
public.ecr.aws/m6i5k1r2/sageworks_dashboard:latest
```
```
docker push public.ecr.aws/m6i5k1r2/sageworks_dashboard:latest
```

### Test the ECR Image
You have a `docker_ecr_dashboard` alias in your `~/.zshrc` :)


