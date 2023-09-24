# Docker Build and Push

Notes and information on how to do the Docker Builds and Push to AWS ECR.

The following instructions should work, but things change :)

### Build the Docker Image
```
docker build -f Dockerfile -t sageworks_dashboard:v0_1_9_amd64 \
--platform linux/amd64 .
```

### Test the Image Locally

### Tag/Push the Image to AWS ECR

### Test the ECR Image


