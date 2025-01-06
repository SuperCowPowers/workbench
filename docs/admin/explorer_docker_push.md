# Explorer Docker Build and Push

Notes and information on how to do the Dashboard Docker Builds and Push to AWS ECR.

### Update Workbench Version
```
cd applications/compound_explorer
vi Dockerfile

# Install Workbench (changes often)
RUN pip install --no-cache-dir workbench==0.8.90 <-- change this
```

### Build the Docker Image
**Note:** For a client specific config file you'll need to copy it locally so that it's within Dockers 'build context'. If you're building the 'vanilla' open source Docker image, then you can use the `open_source_config.json` that's in the directory already.

```
docker build --build-arg WORKBENCH_CONFIG=open_source_config.json -t \
compound_explorer:v0_8_90_amd64 --platform linux/amd64 .
```

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
docker tag compound_explorer:v0_8_90_amd64 \
public.ecr.aws/m6i5k1r2/compound_explorer:v0_8_90_amd64
```
```
docker push public.ecr.aws/m6i5k1r2/compound_explorer:v0_8_90_amd64
```

### Update the 'latest' tag
```
docker tag public.ecr.aws/m6i5k1r2/compound_explorer:v0_8_90_amd64 \
public.ecr.aws/m6i5k1r2/compound_explorer:latest
```
```
docker push public.ecr.aws/m6i5k1r2/compound_explorer:latest
```

### Update the 'stable' tag
This is obviously only when you want to mark a version as stable. Meaning that it seems to 'be good and stable (ish)' :)

```
docker tag public.ecr.aws/m6i5k1r2/compound_explorer:v0_8_90_amd64 \
public.ecr.aws/m6i5k1r2/compound_explorer:stable
```
```
docker push public.ecr.aws/m6i5k1r2/workbench_dashboard:stable
```

### Test the ECR Image
You have a `docker_ecr_dashboard` alias in your `~/.zshrc` :)


