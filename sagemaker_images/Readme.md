## SageMaker Images

Custom Docker images for SageMaker training and inference endpoints.

### Image Types

| Image | Directory | ECR Repository | Description &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |
|-------|-----------|----------------|-------------|
| Base Training | `base/training/` | `aws-ml-images/py312-base-training` | scikit-learn, xgboost, lightgbm, rdkit, workbench-bridges (also used for meta models) |
| Base Inference | `base/inference/` | `aws-ml-images/py312-base-inference` | Same as base training + FastAPI/uvicorn inference server |
| PyTorch/Chem Training | `pytorch_chem/training/` | `aws-ml-images/py312-pytorch-chem-training` | PyTorch, chemprop, rdkit (GPU training via CUDA base image) |
| PyTorch/Chem Inference | `pytorch_chem/inference/` | `aws-ml-images/py312-pytorch-chem-inference` | Same stack as pytorch_chem training + FastAPI/uvicorn (runs on CPU instances) |
| ML Pipelines | `ml_pipelines/` | `aws-ml-images/py312-ml-pipelines` | AWS Batch Pipeline runner for SageMaker ML pipelines |

### Directory Structure

```
sagemaker_images/
  build_deploy.sh          # Build and deploy script
  base/
    training/              # Dockerfile + requirements.txt
    inference/             # Dockerfile + requirements.txt
  pytorch_chem/
    training/              # Dockerfile + requirements.txt (CUDA base image)
    inference/             # Dockerfile + requirements.txt
  ml_pipelines/            # Dockerfile + requirements.txt + runner script
  shared/                  # Files shared across images
    main.py                # FastAPI inference server
    serve                  # SageMaker inference entrypoint
    sagemaker_entrypoint.py # SageMaker training entrypoint
  tests/                   # Container tests
```

### Prerequisites

- Docker installed and running
- AWS CLI configured with appropriate profile
- `AWS_PROFILE` environment variable set (for deploy)

### Build and Deploy

```bash
# From the sagemaker_images/ directory:

# Build only (local)
./build_deploy.sh base/training 0.1

# Build and deploy to ECR (us-east-1 and us-west-2)
./build_deploy.sh base/training 0.1 --deploy

# Overwrite an existing version in ECR
./build_deploy.sh base/training 0.1 --deploy --overwrite
```

#### All image types:
```bash
./build_deploy.sh base/training 0.1 --deploy
./build_deploy.sh base/inference 0.1 --deploy
./build_deploy.sh pytorch_chem/training 0.1 --deploy
./build_deploy.sh pytorch_chem/inference 0.1 --deploy
./build_deploy.sh ml_pipelines 0.1 --deploy
```

### Running Tests

Tests validate containers locally by simulating the SageMaker training and inference environment with Docker.

**Training test** (`test_training.py`): Mounts volumes matching the SageMaker directory layout (`/opt/ml/input`, `/opt/ml/model`, etc.), runs a model script inside the container, and verifies model artifacts are produced.

**Inference test** (`test_inference.py`): Deploys a container with a dummy model, then tests the `/ping` endpoint and both CSV and JSON prediction requests.

```bash
# Test base images (version defaults to 0.1)
./tests/run_tests.sh base

# Test pytorch_chem images with a specific version
./tests/run_tests.sh pytorch_chem 0.2
```

You can also run the Python scripts directly for more control:

```bash
# Test training with a custom model script
python tests/test_training.py \
  --image aws-ml-images/py312-base-training:0.1 \
  --entry-point example_model_script.py \
  --source-dir tests/ \
  --data tests/data/abalone_sm.csv

# Test inference with a specific image
python tests/test_inference.py --image aws-ml-images/py312-base-inference:0.1
```

### Notes

- All images are built for `linux/amd64` (SageMaker runs on x86)
- Deploy pushes both a versioned tag and `latest` to ECR
- The `--overwrite` flag is required to replace an existing versioned tag
- **Training flow**: `sagemaker_entrypoint.py` downloads model script from S3, runs it, then bundles the script + `inference-metadata.json` into the model artifacts
- **Inference flow**: `serve` launches `main.py` (FastAPI), which reads `inference-metadata.json` from the model artifacts to dynamically load the model script's `model_fn`, `input_fn`, `predict_fn`, and `output_fn`
