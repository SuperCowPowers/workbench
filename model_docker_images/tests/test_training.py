#!/usr/bin/env python
import os
import json
import shutil
import argparse
import subprocess
import tempfile
import time
from pathlib import Path


class MockEstimator:
    """
    Mock SageMaker Estimator class that simulates the behavior of sagemaker.estimator.Estimator
    for local testing purposes.
    """

    def __init__(self,
                 image_uri,
                 entry_point=None,
                 source_dir=None,
                 hyperparameters=None,
                 role=None,
                 instance_type=None,
                 **kwargs):
        """
        Initialize a MockEstimator with the same parameters as a real SageMaker Estimator.

        Args:
            image_uri (str): The Docker image URI to use for training
            entry_point (str): The name of the training script
            source_dir (str): Directory with the training script and any additional files
            hyperparameters (dict): Hyperparameters for the training job
            role (str): AWS IAM role (not used in mock)
            instance_type (str): EC2 instance type (not used in mock)
            **kwargs: Additional arguments
        """
        self.image_uri = image_uri
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.hyperparameters = hyperparameters or {}
        self.role = role  # Not used in mock
        self.instance_type = instance_type  # Not used in mock
        self.kwargs = kwargs
        self.temp_dir = None
        self.model_data = None

    def fit(self, inputs, job_name=None, wait=True, logs=True):
        """
        Train the model using the input data.

        Args:
            inputs (dict): Dictionary of input data channels
            job_name (str): Name for the training job
            wait (bool): Whether to wait for the job to complete
            logs (bool): Whether to show the logs

        Returns:
            self: The estimator itself
        """
        print(f"Starting mock training job: {job_name or 'unnamed-job'}")

        try:
            # Create SageMaker directory structure
            self.temp_dir = tempfile.mkdtemp(prefix="sagemaker-test-")
            print(f"Created SageMaker test environment at: {self.temp_dir}")

            # Create the SageMaker directory structure
            os.makedirs(f"{self.temp_dir}/input/data/train", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/input/config", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/model", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/output/data", exist_ok=True)
            os.makedirs(f"{self.temp_dir}/code", exist_ok=True)

            # Process input channels and copy data
            for channel_name, channel_data in inputs.items():
                channel_dir = f"{self.temp_dir}/input/data/{channel_name}"
                os.makedirs(channel_dir, exist_ok=True)

                # Assuming channel_data is a local file path for this mock implementation
                if os.path.isfile(channel_data):
                    shutil.copy2(channel_data, channel_dir)
                    print(f"Copied data file: {channel_data} to {channel_dir}")
                elif os.path.isdir(channel_data):
                    for file in os.listdir(channel_data):
                        if file.endswith(".csv"):
                            shutil.copy2(os.path.join(channel_data, file), channel_dir)
                            print(f"Copied data file: {os.path.join(channel_data, file)} to {channel_dir}")

            # Copy source files to code directory
            if self.source_dir and os.path.exists(self.source_dir):
                # Copy all Python files from source_dir
                for file in os.listdir(self.source_dir):
                    if file.endswith(".py"):
                        shutil.copy2(os.path.join(self.source_dir, file), f"{self.temp_dir}/code")
                        print(f"Copied source file: {os.path.join(self.source_dir, file)} to {self.temp_dir}/code")

            # Prepare hyperparameters.json
            # The key SageMaker parameters
            sagemaker_params = {
                "sagemaker_program": self.entry_point,
                "sagemaker_submit_directory": "/opt/ml/code"  # Container path
            }

            # Combine with user hyperparameters
            all_hyperparams = {**self.hyperparameters, **sagemaker_params}

            # Write the hyperparameters to a JSON file
            with open(f"{self.temp_dir}/input/config/hyperparameters.json", "w") as f:
                json.dump(all_hyperparams, f)

            print(f"Created hyperparameters.json with entry point: {self.entry_point}")

            # Build the Docker command
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.temp_dir}/input:/opt/ml/input",
                "-v", f"{self.temp_dir}/model:/opt/ml/model",
                "-v", f"{self.temp_dir}/output:/opt/ml/output",
                "-v", f"{self.temp_dir}/code:/opt/ml/code",
                "-e", f"SAGEMAKER_PROGRAM={self.entry_point}",
                "-e", "SM_MODEL_DIR=/opt/ml/model",
                "-e", "SM_OUTPUT_DATA_DIR=/opt/ml/output/data",
                "-e", "SM_CHANNEL_TRAIN=/opt/ml/input/data/train"
            ]

            # Add platform flag for Mac M1/M2/M3 users
            if os.uname().machine == 'arm64':
                cmd.insert(2, "--platform")
                cmd.insert(3, "linux/amd64")

            # Add the image URI
            cmd.append(self.image_uri)

            print(f"Running training container with command: {' '.join(cmd)}")

            # Run the container
            start_time = time.time()
            try:
                if logs:
                    # Run with output visible
                    subprocess.run(cmd, check=True)
                else:
                    # Run silently
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                end_time = time.time()
                print(f"Training completed in {end_time - start_time:.2f} seconds")

                # Check the output
                self._check_training_output()

                # Set the model data path (like SageMaker would)
                self.model_data = f"{self.temp_dir}/model"

                return self

            except subprocess.CalledProcessError as e:
                print(f"Error running training container: {e}")
                if e.stdout:
                    print(f"STDOUT: {e.stdout.decode('utf-8')}")
                if e.stderr:
                    print(f"STDERR: {e.stderr.decode('utf-8')}")
                raise

        except Exception as e:
            print(f"Error during fit: {e}")
            raise

    def _check_training_output(self):
        """Check if the training produced output files in the model directory."""
        model_dir = f"{self.temp_dir}/model"
        model_files = os.listdir(model_dir)

        if not model_files:
            print("❌ Warning: No files found in model directory after training")
        else:
            print(f"✅ Found model files: {', '.join(model_files)}")

    def cleanup(self):
        """Remove temporary directories."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


def main():
    """Run the test using a MockEstimator."""
    parser = argparse.ArgumentParser(description="Test SageMaker training container")
    parser.add_argument("--image", type=str, default="aws_model_training:0.1", help="Training image name:tag")
    parser.add_argument("--entry-point", type=str, default="example_model_script.py",
                        help="Name of the training script")
    parser.add_argument("--source-dir", type=str, default="tests/",
                        help="Directory containing the training script")
    parser.add_argument("--data", type=str, default="tests/data/abalone_sm.csv",
                        help="Path to training data file or directory")
    parser.add_argument("--cleanup", action="store_true", help="Clean up temporary files after test")
    args = parser.parse_args()

    # Handle relative paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent

    if not os.path.isabs(args.source_dir):
        args.source_dir = os.path.join(project_root, args.source_dir)

    if not os.path.isabs(args.data):
        args.data = os.path.join(project_root, args.data)

    print(f"Testing with:")
    print(f"  Image: {args.image}")
    print(f"  Entry point: {args.entry_point}")
    print(f"  Source directory: {args.source_dir}")
    print(f"  Training data: {args.data}")

    # Create the estimator
    estimator = MockEstimator(
        image_uri=args.image,
        entry_point=args.entry_point,
        source_dir=args.source_dir,
        # Common SageMaker instance type for training
        instance_type="ml.m5.large"
    )

    try:
        # Run training
        estimator.fit(
            inputs={"train": args.data},
            job_name="mock-training-job"
        )
        print("📋 MockEstimator training completed successfully")

    except Exception as e:
        print(f"❌ MockEstimator training failed: {e}")
        raise

    finally:
        # Clean up if requested
        if args.cleanup:
            estimator.cleanup()
        else:
            print(f"Temporary files are in: {estimator.temp_dir}")
            print("Not removing temporary files for debugging purposes.")


if __name__ == "__main__":
    main()