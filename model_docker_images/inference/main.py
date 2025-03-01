from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import os
import json
import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and metadata
model = None
model_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle model loading on startup and cleanup on shutdown."""
    global model, model_metadata
    model_path = os.environ.get('MODEL_PATH', '/opt/ml/model')
    model_file = os.path.join(model_path, 'model.joblib')

    try:
        logger.info(f"Loading model from {model_path}")

        # Check if model file exists
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            logger.info(f"Model loaded successfully: {type(model)}")
        else:
            # Log the error and available files
            logger.error(f"Model file not found at {model_file}")
            if os.path.exists(model_path):
                logger.error(f"Contents of {model_path}: {os.listdir(model_path)}")
            else:
                logger.error(f"Model directory {model_path} does not exist")

            # Fail fast - no fallback for production
            raise FileNotFoundError(f"Required model file not found: {model_file}")

        # Load metadata if available
        metadata_file = os.path.join(model_path, 'metadata.json')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded model metadata")
        else:
            logger.info(f"No metadata found, using default")
            model_metadata = {'feature_names': None}

    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        # In production, we don't want to create fallback models
        # Let the container fail to start
        raise

    logger.info("Model initialization complete")
    yield
    logger.info("Shutting down model server")


app = FastAPI(lifespan=lifespan)


@app.get('/ping')
def ping():
    """Health check endpoint for SageMaker."""
    if model is not None:
        return Response(status_code=200)
    return Response(status_code=404)


@app.post('/invocations')
async def invoke(request: Request):
    """Inference endpoint for SageMaker."""
    content_type = request.headers.get('Content-Type', '')
    accept_type = request.headers.get('Accept', '')

    try:
        # Get request body
        body = await request.body()

        # Parse input data based on content type
        if 'text/csv' in content_type:
            s = body.decode('utf-8')
            data = pd.read_csv(pd.StringIO(s), header=None)
        else:  # Default to JSON
            json_str = body.decode('utf-8')
            data_json = json.loads(json_str)
            data = pd.DataFrame(data_json) if not isinstance(data_json, pd.DataFrame) else data_json

        # Make prediction
        predictions = model.predict(data)

        # Format response based on accept type
        if 'text/csv' in accept_type:
            result = pd.DataFrame(predictions).to_csv(header=False, index=False)
            return Response(content=result, media_type='text/csv')
        else:  # Default to JSON
            result = json.dumps({
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else float(predictions)
            })
            return Response(content=result, media_type='application/json')

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )
