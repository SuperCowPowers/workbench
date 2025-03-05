# Workbench Base Docker Build and Push

Notes and information on how to do the Docker Builds and Push to AWS ECR.

### Update Workbench Version
```
vi Dockerfile

ARG WORKBENCH_VERSION=0.8.110 <-- change this in TWO places
```

### Run the Deploy Script
**Note:** For a client specific config file you'll need to copy it locally so that it's within Dockers 'build context'. If you're building the 'vanilla' open source Docker image, then you can use the `open_source_config.json` that's in the directory already.

```
./deploy_docker.sh
```

### Update the 'stable' tag
This is obviously only when you want to mark a version as stable. Meaning that it seems to 'be good and stable (ish)' :)

```
./deploy.sh --stable
```

### Test the ECR Image
You have a `docker_ecr_base` alias in your `~/.zshrc` :)


