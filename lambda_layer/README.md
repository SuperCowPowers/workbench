# Building a Lambda Layer for SageWorks

### Step 1: Build the Docker Container

```bash
docker build -t sageworks-lambda-layer .
```

### Step 2: Run the Docker Container with an Interactive Shell

```bash
docker run -it --entrypoint /bin/bash
```

### Step 3: Verify the Installation

Inside the container, check the contents of the `/asset-output/python` directory:

```bash
ls -l /asset-output/python
```

Ensure the `sageworks` package and its dependencies are installed correctly.

### Step 4: Exit the Container

Exit the interactive shell:

```bash
exit
```

### Step 5: Copy the Results to Your Local Disk

First, find the container ID:

```bash
docker ps -a
```

Locate the container ID of the `sageworks-lambda-layer` container, then copy the contents:

```bash
docker cp <container_id>:/asset-output/python ./sageworks-lambda-layer-output/
```
