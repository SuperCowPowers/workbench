import os
import json
import argparse
import tempfile
import shutil
import subprocess
import numpy as np
import pandas as pd


def create_test_data(data_dir, rows=100, cols=5):
    """Create synthetic training data for testing."""
    print(f"Creating synthetic training data in {data_dir}")

    # Generate synthetic features and target
    X = np.random.randn(rows, cols)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + 0.5 * X[:, 3] - X[:, 4] + np.random.randn(rows) * 0.1

    # Create dataframe
    cols = [f"feature_{i}" for i in range(cols)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y

    # Create train directory
    train_dir = os.path.join(data_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Save to CSV
    train_file = os.path.join(train_dir, 'train.csv')
    df.to_csv(train_file, index=False)
    print(f"Saved {rows} rows of training data to {train_file}")

    return train_file


def create_hyperparameters(config_dir):
    """Create hyperparameters.json file for the training container."""
    print(f"Creating hyperparameters in {config_dir}")

    # Define hyperparameters
    hyperparameters = {
        "max_depth": "6",
        "learning_rate": "0.1",
        "n_estimators": "100",
        "objective": "reg:squarederror"
    }

    # Create config directory
    os.makedirs(config_dir, exist_ok=True)

    # Save hyperparameters
    hyperparameters_file = os.path.join(config_dir, 'hyperparameters.json')
    with open(hyperparameters_file, 'w') as f:
        json.dump(hyperparameters, f)

    print(f"Saved hyperparameters to {hyperparameters_file}")
    return hyperparameters_file


def test_training_container(image_name, temp_dir):
    """Run the training container with test data and verify outputs."""
    print(f"\n🔬 Testing training container: {image_name}")

    # Create directory structure to mimic SageMaker
    input_dir = os.path.join(temp_dir, 'input')
    data_dir = os.path.join(input_dir, 'data')
    config_dir = os.path.join(input_dir, 'config')
    model_dir = os.path.join(temp_dir, 'model')
    output_dir = os.path.join(temp_dir, 'output')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Create test data and hyperparameters
    create_test_data(data_dir)
    create_hyperparameters(config_dir)

    # Run the container
    print("\n📦 Running training container...")

    cmd = [
        "docker", "run",
        "--rm",
        "-v", f"{temp_dir}:/opt/ml",
        image_name
    ]

    try:
        # Execute the training container
        subprocess.run(cmd, check=True)

        # Check if model files were created
        model_files = os.listdir(model_dir)
        if not model_files:
            print("❌ Training failed: No model files created")
            return False

        print(f"✅ Training succeeded! Model files created: {', '.join(model_files)}")

        # Check for specific expected files
        expected_files = ['model.joblib', 'metadata.json']
        missing_files = [f for f in expected_files if f not in model_files]

        if missing_files:
            print(f"⚠️ Warning: Some expected files are missing: {', '.join(missing_files)}")
        else:
            print("✅ All expected model files were created")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error code {e.returncode}")

        # Check if there's a failure file with more details
        failure_file = os.path.join(output_dir, 'failure')
        if os.path.exists(failure_file):
            with open(failure_file, 'r') as f:
                failure_content = f.read()
            print(f"Error details:\n{failure_content}")

        return False


def run_training_test(image_name="aws_model_training:latest"):
    """Run the training container test with a temporary directory."""
    print("🚀 Starting training container test")

    # Create temporary directory for training data
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        success = test_training_container(image_name, temp_dir)

    if success:
        print("\n🎉 Training container test passed!")
    else:
        print("\n❌ Training container test failed!")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the AWS model training container")
    parser.add_argument("--image", default="aws_model_training:latest",
                        help="Docker image name for the training container")

    args = parser.parse_args()
    run_training_test(args.image)