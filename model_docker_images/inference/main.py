from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
import os
import json
import numpy as np
import pandas as pd
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model will be accessible globally
model = None
model_metadata = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, model_metadata

    # SageMaker model path
    model_path = os.environ.get('MODEL_PATH', '/opt/ml/model')

    try:
        logger.info(f"Loading model from {model_path}")
        model_file = os.path.join(model_path, 'model.joblib')

        # Check if model file exists
        if not os.path.exists(model_file):
            logger.warning(f"Model file not found at {model_file}")
            # List directory contents for debugging
            if os.path.exists(model_path):
                logger.info(f"Contents of {model_path}: {os.listdir(model_path)}")
            else:
                logger.warning(f"Model directory {model_path} not found")

            # For testing only - create a dummy model
            logger.warning("Creating a dummy model for testing")
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.fit(np.array([[1, 2, 3]]), np.array([1]))
        else:
            # Load the actual model
            logger.info(f"Loading model from {model_file}")
            model = joblib.load(model_file)
            logger.info(f"Model loaded successfully: {type(model)}")

        # Load metadata if available
        try:
            metadata_file = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    model_metadata = json.load(f)
                logger.info(f"Loaded model metadata: {model_metadata}")
            else:
                logger.warning(f"Metadata file not found at {metadata_file}")
                model_metadata = {'feature_names': None}
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            model_metadata = {'feature_names': None}
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        # Provide a fallback model for testing
        import xgboost as xgb
        model = xgb.XGBRegressor()
        model.fit(np.array([[1, 2, 3]]), np.array([1]))
        model_metadata = {'feature_names': None}

    logger.info("Model initialization complete")
    yield

    # Cleanup on shutdown if needed
    logger.info("Cleaning up resources")


app = FastAPI(lifespan=lifespan)


@app.get('/ping')
def ping():
    # SageMaker health check - return 200 if model is loaded
    if model is not None:
        return Response(status_code=200)
    return Response(status_code=404)


@app.post('/invocations')
async def invoke(request: Request):
    logger.info("Received inference request")
    content_type = request.headers.get('Content-Type', '')
    accept_type = request.headers.get('Accept', '')

    logger.info(f"Content-Type: {content_type}, Accept: {accept_type}")

    # Get the data
    body = await request.body()

    try:
        # Handle different content types
        if content_type == 'text/csv':
            # Parse CSV data
            s = body.decode('utf-8')
            data = pd.read_csv(pd.StringIO(s), header=None)
            logger.info(f"Parsed CSV data with shape: {data.shape}")
        else:
            # Default to JSON
            json_str = body.decode('utf-8')
            logger.info(f"Raw JSON input: {json_str}")
            data_json = json.loads(json_str)
            logger.info(f"Parsed JSON data: {data_json}")
            # Convert to DataFrame if it's not already
            if not isinstance(data_json, pd.DataFrame):
                data = pd.DataFrame(data_json)
            else:
                data = data_json

        # Make prediction
        logger.info(f"Making prediction with data shape: {data.shape}")
        predictions = model.predict(data)
        logger.info(f"Prediction successful, result shape: {len(predictions) if hasattr(predictions, '__len__') else 'scalar'}")

        # Always return JSON unless explicitly requested as CSV
        if accept_type == 'text/csv':
            result = pd.DataFrame(predictions).to_csv(header=False, index=False)
            logger.info(f"Returning CSV response: {result}")
            return Response(content=result, media_type='text/csv')
        else:
            # Default to JSON for everything else
            result = json.dumps({'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else float(predictions)})
            logger.info(f"Returning JSON response: {result}")
            return Response(content=result, media_type='application/json')

    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )
