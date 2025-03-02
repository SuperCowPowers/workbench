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
    """Mock SageMaker Estimator for local container testing"""

    def __init__(self, image_uri, entry_point=None, source_dir=None, hyperparameters=None, **kwargs):
        self.image_uri = image_uri
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.hyperparameters = hyperparameters or {}
        self.temp_dir = None
        self.model_data = None

    def fit(self, inputs, job_name=None, logs=True):
        """Train the model using the input data"""
        print(f"Starting mock training job: {job_name or 'unnamed-job'}")

        try:
            # Set up SageMaker directory structure
            self.temp_dir = tempfile.mkdtemp(prefix="sagemaker-test-")
            print(f"Created test environment at: {self.temp_dir}")

            # Create directories
            for path in ["input/data/train", "input/config", "model", "output/data", "code"]:
                os.makedirs(f"{self.temp_dir}/{path}", exist_ok=True)

            # Copy data files
            for channel_name, channel_data in inputs.items():
                channel_dir = f"{self.temp_dir}/input/data/{channel_name}"
                os.makedirs(channel_dir, exist_ok=True)

                if os.path.isfile(channel_data):
                    shutil.copy2(channel_data, channel_dir)
                    print(f"Copied data: {os.path.basename(channel_data)} to {channel_name} channel")
                elif os.path.isdir(channel_data):
                    for file in os.listdir(channel_data):
                        if file.endswith(".csv"):
                            shutil.copy2(os.path.join(channel_data, file), channel_dir)

            # Copy source files to code directory
            if self.source_dir and os.path.exists(self.source_dir):
                for file in os.listdir(self.source_dir):
                    if file.endswith(".py"):
                        shutil.copy2(os.path.join(self.source_dir, file), f"{self.temp_dir}/code")
                print(f"Copied source files to code directory")

            # Create hyperparameters.json
            all_hyperparams = {
                **self.hyperparameters,
                "sagemaker_program": self.entry_point,
                "sagemaker_submit_directory": "/opt/ml/code",
            }

            with open(f"{self.temp_dir}/input/config/hyperparameters.json", "w") as f:
                json.dump(all_hyperparams, f)

            # Run the container
            cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{self.temp_dir}/input:/opt/ml/input",
                "-v",
                f"{self.temp_dir}/model:/opt/ml/model",
                "-v",
                f"{self.temp_dir}/output:/opt/ml/output",
                "-v",
                f"{self.temp_dir}/code:/opt/ml/code",
                "-e",
                f"SAGEMAKER_PROGRAM={self.entry_point}",
                "-e",
                "SM_MODEL_DIR=/opt/ml/model",
                "-e",
                "SM_OUTPUT_DATA_DIR=/opt/ml/output/data",
                "-e",
                "SM_CHANNEL_TRAIN=/opt/ml/input/data/train",
                self.image_uri,
            ]

            # Add platform flag for Mac M1/M2/M3 users
            if os.uname().machine == "arm64":
                cmd.insert(2, "--platform")
                cmd.insert(3, "linux/amd64")

            print(f"Running training container...")

            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=not logs)
            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

            # Check output
            model_files = os.listdir(f"{self.temp_dir}/model")
            if model_files:
                print(f"✅ Model created successfully with files: {', '.join(model_files)}")
            else:
                print("⚠️ No model files were created during training")

            return self

        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with exit code {e.returncode}")
            if e.stdout:
                print(f"STDOUT: {e.stdout.decode('utf-8')}")
            if e.stderr:
                print(f"STDERR: {e.stderr.decode('utf-8')}")
            raise
        except Exception as e:
            print(f"❌ Error during training: {e}")
            raise

    def cleanup(self):
        """Remove temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


def main():
    """Run the test using a MockEstimator"""
    parser = argparse.ArgumentParser(description="Test SageMaker training container")
    parser.add_argument(
        "--image", type=str, default="aws-ml-images/py312-sklearn-xgb-training:0.1", help="Training image name:tag"
    )
    parser.add_argument("--entry-point", type=str, default="example_model_script.py", help="Training script name")
    parser.add_argument("--source-dir", type=str, default="tests/", help="Directory containing training scripts")
    parser.add_argument("--data", type=str, default="tests/data/abalone_sm.csv", help="Training data path")
    args = parser.parse_args()

    # Resolve relative paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent

    source_dir = os.path.join(project_root, args.source_dir) if not os.path.isabs(args.source_dir) else args.source_dir
    data_path = os.path.join(project_root, args.data) if not os.path.isabs(args.data) else args.data

    print(f"Testing with image {args.image}, script {args.entry_point}")

    # Create and run the estimator
    estimator = MockEstimator(image_uri=args.image, entry_point=args.entry_point, source_dir=source_dir)

    try:
        estimator.fit(inputs={"train": data_path}, job_name="mock-training-job")
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise
    finally:
        estimator.cleanup()


if __name__ == "__main__":
    main()
