# Docker Build and Push

Notes and information on how to do the Docker Builds and Push to AWS ECR.

The following instructions should work, but things change :)

### Update SageWorks Version
```
cd applications/aws_dashboard
vi Dockerfile
# Install Sageworks (changes often)
RUN pip install --no-cache-dir sageworks==0.1.14
```

### Build the Docker Image
```
docker build -f Dockerfile -t sageworks_dashboard:v0_1_9_amd64 \
--platform linux/amd64 .
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
docker tag sageworks_dashboard:v0_1_9_amd64 \
public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_1_9_amd64
```
```
docker push public.ecr.aws/m6i5k1r2/sageworks_dashboard:v0_1_9_amd64
```

### Test the ECR Image
You have a `docker_ecr_dashboard` alias in your `~/.zshrc` :)


