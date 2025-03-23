# Dashboard Docker Build and Push

Notes and information on how to do the Dashboard Docker Builds and Push to AWS ECR.

### Update Workbench Version
```
cd applications/aws_dashboard
vi Dockerfile

# Install workbench[ui] in separate layer
RUN pip install --no-cache-dir "workbench[ui]==0.8.116"  <-- change this
```

### Run the Deploy Script
**Note:** For a client specific config file you'll need to copy it locally so that it's within Dockers 'build context'. If you're building the 'vanilla' open source Docker image, then you can use the `open_source_config.json` that's in the directory already.

```
./deploy.sh 0_8_110   (match version to above)
```

**Custom Plugins:** If you're using custom plugins those are simply 'pushed' to S3, visit our [Dashboard with Plugins](dashboard_s3_plugins.md)) page.

### Update the 'stable' tag
This is obviously only when you want to mark a version as stable. Meaning that it seems to 'be good and stable (ish)' :)

```
./deploy.sh 0_8_110 --stable
```

### Test the ECR Image
You have a `docker_ecr_dashboard` alias in your `~/.zshrc` :)


