from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import os
import sys
import json
import importlib.util
import logging
import subprocess
import site

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
inference_module = None


def get_inference_script(model_dir: str) -> str:
    """Retrieve the inference script name

    Args:
        model_dir (str): The directory containing the model artifacts

    Returns:
        str: The name of the inference script
    """

    # Get the path to the inference-metadata.json file
    inference_meta_path = os.path.join(model_dir, "inference-metadata.json")
    with open(inference_meta_path, "r") as f:
        config = json.load(f)
        return config["inference_script"]


def install_requirements(requirements_path):
    """Install Python dependencies from requirements file.
    Uses a persistent cache to speed up container cold starts.
    Note: Inference containers don't have root access, so we
          use the --user flag and add the user package path manually.
    """
    if os.path.exists(requirements_path):
        logger.info(f"Installing dependencies from {requirements_path}...")

        # Define a persistent cache location
        pip_cache_dir = "/opt/ml/model/.cache/pip"
        os.environ["PIP_CACHE_DIR"] = pip_cache_dir

        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--cache-dir",
                    pip_cache_dir,  # Enable caching
                    "--disable-pip-version-check",
                    "--no-warn-script-location",
                    "--user",
                    "-r",
                    requirements_path,
                ]
            )
            # Ensure Python can find user-installed packages
            sys.path.append(site.getusersitepackages())
            logger.info("Requirements installed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing requirements: {e}")
            sys.exit(1)
    else:
        logger.info(f"No requirements file found at {requirements_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle model loading on startup and cleanup on shutdown."""
    global model, inference_module

    # Note: SageMaker will put model.tar.gz in /opt/ml/model
    #       which includes the model artifacts and inference code
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    inference_script = get_inference_script(model_dir)

    # List directory contents for debugging
    logger.info(f"Contents of {model_dir}: {os.listdir(model_dir)}")

    try:
        # Load the inference script from source_dir
        inference_script_path = os.path.join(model_dir, inference_script)
        if not os.path.exists(inference_script_path):
            raise FileNotFoundError(f"Inference script not found: {inference_script_path}")

        # Install requirements if present
        install_requirements(os.path.join(model_dir, "requirements.txt"))

        # Ensure the model directory is in the Python path
        sys.path.insert(0, model_dir)

        # Import the inference module
        logger.info(f"Importing inference module from {inference_script_path}")
        spec = importlib.util.spec_from_file_location("inference_module", inference_script_path)
        inference_module = importlib.util.module_from_spec(spec)
        sys.modules["inference_module"] = inference_module
        spec.loader.exec_module(inference_module)

        # Check if model_fn is defined
        if not hasattr(inference_module, "model_fn"):
            raise ImportError(f"Inference module {inference_script_path} does not define model_fn")

        # Load the model using model_fn
        logger.info("Calling model_fn to load the model")
        model = inference_module.model_fn(model_dir)
        logger.info(f"Model loaded successfully: {type(model)}")

    except Exception as e:
        logger.error(f"Error initializing model: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down model server")


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    """Health check endpoint for SageMaker."""
    # Check if the inference module is loaded
    return Response(status_code=200 if inference_module else 500)


@app.post("/invocations")
async def invoke(request: Request):
    """Inference endpoint for SageMaker."""
    content_type = request.headers.get("Content-Type", "")
    accept_type = request.headers.get("Accept", "")

    try:
        body = await request.body()
        data = inference_module.input_fn(body, content_type)
        result = inference_module.predict_fn(data, model)
        output_data, output_content_type = inference_module.output_fn(result, accept_type)
        return Response(content=output_data, media_type=output_content_type)
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return Response(content=json.dumps({"error": str(e)}), status_code=500, media_type="application/json")
