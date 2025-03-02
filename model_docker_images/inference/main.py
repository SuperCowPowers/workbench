from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import os
import sys
import json
import importlib.util
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
inference_module = None


def get_inference_script():
    """Retrieve the entry point script name for SageMaker inference."""
    # Check SAGEMAKER_PROGRAM first
    if "SAGEMAKER_PROGRAM" in os.environ:
        return os.environ["SAGEMAKER_PROGRAM"]

    # For inference containers, check these common locations
    model_server_config = "/opt/ml/model/model-config.json"
    if os.path.exists(model_server_config):
        try:
            with open(model_server_config, "r") as f:
                config = json.load(f)
                if "inference_script" in config:
                    return config["inference_script"]
        except Exception as e:
            print(f"Error reading model-config.json: {e}")

    # Debug available environment variables
    print("Available environment variables:")
    for key in os.environ:
        print(f"  {key}: {os.environ[key]}")

    # Recursively list out all files in /opt/ml
    print("Contents of /opt/ml:")
    for root, dirs, files in os.walk("/opt/ml"):
        for file in files:
            print(f"  {root}/{file}")


def get_model_script():
    """Retrieve the SAGEMAKER_PROGRAM from environment variable or hyperparameters.json."""
    if "SAGEMAKER_PROGRAM" in os.environ:
        return os.environ["SAGEMAKER_PROGRAM"]

    # Look for hyperparameters.json
    hyperparams_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(hyperparams_path):
        try:
            with open(hyperparams_path, "r") as f:
                hyperparams = json.load(f)
                if "sagemaker_program" in hyperparams:
                    return hyperparams["sagemaker_program"]
        except Exception as e:
            print(f"Error reading hyperparameters.json: {e}")

    # If no program is found, raise an error
    raise ValueError("SAGEMAKER_PROGRAM not found in environment variables or hyperparameters.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle model loading on startup and cleanup on shutdown."""
    global model, inference_module

    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    code_dir = os.environ.get("SM_MODULE_DIR", "/opt/ml/code")

    # Add code_dir to sys.path so that any local utilities can be imported
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    model_script = get_inference_script()

    try:
        logger.info(f"Loading model from {model_dir}")
        logger.info(f"Loading inference code from {code_dir}")

        # Ensure directories exist
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        if not os.path.exists(code_dir):
            raise FileNotFoundError(f"Code directory not found: {code_dir}")

        # List directory contents for debugging
        logger.info(f"Contents of {model_dir}: {os.listdir(model_dir)}")
        logger.info(f"Contents of {code_dir}: {os.listdir(code_dir)}")

        # Load the inference module from source_dir
        entry_point_path = os.path.join(code_dir, model_script)
        if not os.path.exists(entry_point_path):
            raise FileNotFoundError(f"Entry point script {model_script} not found in {code_dir}")

        logger.info(f"Importing inference module from {entry_point_path}")
        spec = importlib.util.spec_from_file_location("inference_module", entry_point_path)
        inference_module = importlib.util.module_from_spec(spec)
        sys.modules["inference_module"] = inference_module
        spec.loader.exec_module(inference_module)

        if not hasattr(inference_module, "model_fn"):
            raise ImportError(f"Inference module {model_script} does not define model_fn")

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
    return Response(status_code=200 if model else 404)


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
