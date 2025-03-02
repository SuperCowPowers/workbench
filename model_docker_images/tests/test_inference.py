#!/usr/bin/env python
import os
import json
import time
import argparse
import tempfile
import shutil
import subprocess
import requests
import pandas as pd
import numpy as np
from io import StringIO


class MockModel:
    """Mock SageMaker Model class that simulates the behavior of sagemaker.model.Model"""

    def __init__(self, image_uri, model_data=None, role=None, **kwargs):
        """
        Initialize a MockModel with parameters similar to a SageMaker Model.

        Args:
            image_uri (str): The Docker image URI to use for inference
            model_data (str): Path to model artifacts (S3 URI or local path)
            role (str): AWS IAM role (not used in mock)
        """
        self.image_uri = image_uri
        self.model_data = model_data
        self.role = role
        self.kwargs = kwargs
        self.temp_dir = None
        self.container_id = None
        self.endpoint_url = None

    def register(self, content_types=None, response_types=None, **kwargs):
        """Mock model registration - just stores the parameters"""
        self.content_types = content_types or ["application/json"]
        self.response_types = response_types or ["application/json"]
        for key, value in kwargs.items():
            setattr(self, key, value)
        print(f"Mock registered model with content types: {self.content_types}")
        return self

    def deploy(self, instance_type=None, initial_instance_count=1, endpoint_name=None):
        """
        Deploy the model to a mock endpoint (local Docker container).

        Args:
            instance_type (str): SageMaker instance type (ignored)
            initial_instance_count (int): Number of instances (ignored)
            endpoint_name (str): Endpoint name for identification

        Returns:
            MockEndpoint: The deployed endpoint
        """
        print(f"Deploying model to endpoint: {endpoint_name or 'default-endpoint'}")

        # Create a temp directory for model data if not provided
        if self.model_data is None:
            self.temp_dir = tempfile.mkdtemp(prefix="sagemaker-inference-test-")
            model_dir = self.temp_dir

            # Create a dummy model
            print(f"Creating dummy model in {model_dir}")
            import joblib
            import xgboost as xgb

            # Train a simple model
            model = xgb.XGBRegressor(objective="reg:squarederror")
            X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            y = np.array([10, 20, 30])
            model.fit(X, y)

            # Save the model
            joblib.dump(model, os.path.join(model_dir, "model.joblib"))

            # Save metadata
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump({"feature_names": ["feature1", "feature2", "feature3"], "model_type": "regression"}, f)

            self.model_data = model_dir
        else:
            # Use provided model_data
            model_dir = self.model_data

        # Start the container
        cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            "8080:8080",
            "-v",
            f"{model_dir}:/opt/ml/model",
            "-e",
            "MODEL_PATH=/opt/ml/model",
        ]

        # Add platform flag for Mac M1/M2/M3 users
        if os.uname().machine == "arm64":
            cmd.insert(2, "--platform")
            cmd.insert(3, "linux/amd64")

        # Add the image URI
        cmd.append(self.image_uri)
        print(f"Starting inference container: {' '.join(cmd)}")
        self.container_id = subprocess.check_output(cmd).decode("utf-8").strip()

        # Add this block immediately after starting the container
        print(f"Container ID: {self.container_id}")
        try:
            # Give it a moment to start or fail
            time.sleep(1)

            # Get container logs
            logs = subprocess.check_output(["docker", "logs", self.container_id], stderr=subprocess.STDOUT).decode(
                "utf-8"
            )
            print(f"Container startup logs:\n{logs}")
        except Exception as e:
            print(f"Error getting container logs: {e}")

        self.endpoint_url = "http://localhost:8080"
        return MockEndpoint(self)


class MockEndpoint:
    """Mock SageMaker Endpoint for local testing"""

    def __init__(self, model):
        """Initialize with a reference to the model"""
        self.model = model
        self.url = model.endpoint_url

        # Check container status and logs
        try:
            # Get container state
            inspect_output = (
                subprocess.check_output(["docker", "inspect", "--format", "{{.State.Status}}", model.container_id])
                .decode("utf-8")
                .strip()
            )

            print(f"Container status: {inspect_output}")

            # If not running, get the logs
            if inspect_output != "running":
                logs = subprocess.check_output(["docker", "logs", model.container_id], stderr=subprocess.STDOUT).decode(
                    "utf-8"
                )
                print(f"Container logs:\n{logs}")
                raise RuntimeError("Container failed to start properly")
        except Exception as e:
            print(f"Error checking container: {e}")

    def predict(self, data, initial_args=None):
        """
        Makes a prediction using the deployed model.

        Args:
            data: Input data in format matching content_types
            initial_args: Additional arguments (ignored)

        Returns:
            The prediction result
        """
        # Default to first registered content type
        content_type = self.model.content_types[0] if hasattr(self.model, "content_types") else "application/json"

        # Format the data according to content type
        if content_type == "text/csv":
            if isinstance(data, pd.DataFrame):
                payload = data.to_csv(header=False, index=False)
            elif isinstance(data, (list, np.ndarray)):
                payload = pd.DataFrame(data).to_csv(header=False, index=False)
            else:
                payload = str(data)
        else:
            # Default to JSON
            if isinstance(data, pd.DataFrame):
                payload = data.to_json(orient="records")
            elif isinstance(data, (list, np.ndarray)):
                payload = json.dumps({"instances": data.tolist() if hasattr(data, "tolist") else data})
            else:
                payload = json.dumps(data)

        # Send the request to the container
        try:
            response = requests.post(f"{self.url}/invocations", data=payload, headers={"Content-Type": content_type})

            # Check for errors
            if response.status_code != 200:
                raise Exception(f"Prediction failed with status code {response.status_code}: {response.text}")

            # Parse response based on response type
            if hasattr(self.model, "response_types") and "text/csv" in self.model.response_types:
                # Parse CSV response
                return pd.read_csv(StringIO(response.text), header=None)
            else:
                # Parse JSON response
                return response.json()

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def delete_endpoint(self):
        """Clean up resources by stopping the container"""
        print(f"Deleting endpoint (stopping container {self.model.container_id})")
        if self.model.container_id:
            try:
                subprocess.run(["docker", "stop", self.model.container_id], check=False)
            except Exception as e:
                print(f"Error stopping container: {e}")

        # Clean up temp directory if needed
        if self.model.temp_dir and os.path.exists(self.model.temp_dir):
            print(f"Cleaning up temporary directory: {self.model.temp_dir}")
            shutil.rmtree(self.model.temp_dir)
            self.model.temp_dir = None


def test_csv_inference(endpoint, test_data=None):
    """Test inference with CSV data"""
    print("\nTesting CSV inference...")

    if test_data is None:
        # Create sample test data
        test_data = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    try:
        response = endpoint.predict(test_data)
        print(f"Prediction response: {response}")
        print("✅ CSV inference test successful")
        return True
    except Exception as e:
        print(f"❌ CSV inference test failed: {e}")
        return False


def test_json_inference(endpoint, test_data=None):
    """Test inference with JSON data"""
    print("\nTesting JSON inference...")

    if test_data is None:
        # Create sample test data - use list of lists of floats
        test_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    try:
        response = endpoint.predict(test_data)
        print(f"Prediction response: {response}")
        print("✅ JSON inference test successful")
        return True
    except Exception as e:
        print(f"❌ JSON inference test failed: {e}")
        return False


def test_ping_endpoint(url):
    """Test the /ping endpoint directly"""
    print("\nTesting /ping endpoint...")
    try:
        response = requests.get(f"{url}/ping")
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Ping test successful")
            return True
        else:
            print(f"❌ Ping test failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ping test error: {e}")
        return False


def main():
    """Run the test using MockModel and MockEndpoint"""
    parser = argparse.ArgumentParser(description="Test SageMaker inference container")
    parser.add_argument(
        "--image", type=str, default="aws-ml-images/py312-sklearn-xgb-inference:0.1", help="Inference image name:tag"
    )
    parser.add_argument("--model-dir", type=str, default=None, help="Path to model directory (optional)")
    args = parser.parse_args()

    print(f"Testing inference container {args.image}")

    # Create the model and endpoint
    model = None
    endpoint = None
    success = False

    try:
        # Create and deploy the model
        model = MockModel(image_uri=args.image, model_data=args.model_dir, role="mock-role")

        # Register the model
        model.register(
            content_types=["text/csv", "application/json"],
            response_types=["text/csv", "application/json"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            description="Test model",
        )

        # Deploy the model
        endpoint = model.deploy(instance_type="local", initial_instance_count=1, endpoint_name="test-endpoint")

        # Test the /ping endpoint
        ping_success = test_ping_endpoint(endpoint.url)

        # Test predictions
        csv_success = test_csv_inference(endpoint)
        json_success = test_json_inference(endpoint)

        # Overall success
        success = ping_success and csv_success and json_success

        if success:
            print("\n✅ All inference tests passed successfully!")
        else:
            print("\n❌ Some inference tests failed!")

    except Exception as e:
        print(f"\n❌ Error during inference testing: {e}")
    finally:
        # Clean up resources
        if endpoint:
            endpoint.delete_endpoint()

    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
