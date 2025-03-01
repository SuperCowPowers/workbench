import os
import json
import shutil
import argparse
import subprocess
import tempfile
import time
from pathlib import Path


def setup_sagemaker_directories():
    """Create a temporary directory structure that mimics SageMaker's layout."""
    base_dir = tempfile.mkdtemp(prefix="sagemaker-test-")

    # Create the SageMaker directory structure
    os.makedirs(f"{base_dir}/input/data/train", exist_ok=True)
    os.makedirs(f"{base_dir}/input/config", exist_ok=True)
    os.makedirs(f"{base_dir}/model", exist_ok=True)
    os.makedirs(f"{base_dir}/output/data", exist_ok=True)
    os.makedirs(f"{base_dir}/code", exist_ok=True)

    return base_dir


def copy_sample_data(base_dir, data_file):
    """Copy sample data to the training directory."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Sample data file not found: {data_file}")

    shutil.copy2(data_file, f"{base_dir}/input/data/train/")
    print(f"Copied sample data: {data_file} to {base_dir}/input/data/train/")


def copy_model_script(base_dir, script_file):
    """Copy the model script to the code directory."""
    if not os.path.exists(script_file):
        raise FileNotFoundError(f"Model script not found: {script_file}")

    shutil.copy2(script_file, f"{base_dir}/code/")
    print(f"Copied model script: {script_file} to {base_dir}/code/")

    return os.path.basename(script_file)


def create_hyperparameters(base_dir, script_name, hyperparams=None):
    """Create a hyperparameters.json file with SageMaker-specific entries."""
    if hyperparams is None:
        hyperparams = {}

    # Add required SageMaker hyperparameters
    hyperparams["sagemaker_program"] = script_name
    hyperparams["sagemaker_submit_directory"] = "/opt/ml/code"

    # Write the hyperparameters to a JSON file
    with open(f"{base_dir}/input/config/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f)

    print(f"Created hyperparameters.json with script: {script_name}")


def run_training_container(base_dir, image_name, script_name):
    """Run the training container with the proper volume mounts and environment variables."""
    # Build the Docker command
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{base_dir}/input:/opt/ml/input",
        "-v", f"{base_dir}/model:/opt/ml/model",
        "-v", f"{base_dir}/output:/opt/ml/output",
        "-v", f"{base_dir}/code:/opt/ml/code",
        "-e", f"SAGEMAKER_PROGRAM={script_name}",
        "-e", "SM_MODEL_DIR=/opt/ml/model",
        "-e", "SM_OUTPUT_DATA_DIR=/opt/ml/output/data",
        "-e", "SM_CHANNEL_TRAIN=/opt/ml/input/data/train",
        image_name
    ]

    print(f"Running training container with command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        subprocess.run(cmd, check=True)
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running training container: {e}")
        return False


def check_training_output(base_dir):
    """Check if the training produced the expected output files."""
    model_dir = f"{base_dir}/model"
    output_dir = f"{base_dir}/output"

    # Check if model files were created
    model_files = os.listdir(model_dir)
    print(f"Files in model directory: {model_files}")

    # Check for xgb_model.json which should be created by our example script
    if "xgb_model.json" in model_files and "feature_columns.json" in model_files:
        print("✅ Training successful! Model files were created.")
        return True
    else:
        print("❌ Training failed! Expected model files were not created.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test SageMaker training container")
    parser.add_argument("--image", type=str, required=True, help="Training image name:tag")
    parser.add_argument("--script", type=str, default="example_model_script.py",
                        help="Path to the model script to test")
    parser.add_argument("--data", type=str, default="tests/data/abalone_sm.csv",
                        help="Path to sample data file")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent

    if not os.path.isabs(args.script):
        args.script = os.path.join(script_dir, args.script)

    if not os.path.isabs(args.data):
        args.data = os.path.join(project_root, args.data)

    try:
        # Setup the SageMaker-like directory structure
        base_dir = setup_sagemaker_directories()
        print(f"Created SageMaker test environment at: {base_dir}")

        # Copy the sample data
        copy_sample_data(base_dir, args.data)

        # Copy the model script and get its basename
        script_name = copy_model_script(base_dir, args.script)

        # Create hyperparameters.json
        # You could add more hyperparameters here specific to your model
        hyperparams = {
            "model_type": "regressor",
            "target_column": "rings",
            "feature_list": '["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]',
            "train_all_data": "False"
        }
        create_hyperparameters(base_dir, script_name, hyperparams)

        # Run the training container
        success = run_training_container(base_dir, args.image, script_name)

        if success:
            # Check if training produced expected output
            check_training_output(base_dir)

        # Cleanup
        print(f"Temporary files are in: {base_dir}")
        print("Not removing temporary files for debugging purposes.")
        # If you want to auto-cleanup, uncomment the following line:
        # shutil.rmtree(base_dir)

    except Exception as e:
        print(f"Error during test: {e}")
        raise


if __name__ == "__main__":
    main()
