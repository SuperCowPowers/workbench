import os
import json
import sys
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# SageMaker paths
prefix = '/opt/ml/'
input_path = prefix + 'input/data'
model_path = os.path.join(prefix, 'model')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')
output_path = os.path.join(prefix, 'output')

# Channel names for training and validation data
training_channel_name = 'train'
eval_channel_name = 'validation'


# Load hyperparameters
def load_hyperparameters():
    with open(param_path, 'r') as tc:
        hyperparameters = json.load(tc)

    # Convert hyperparameters from strings to appropriate types
    processed_params = {}
    for key, value in hyperparameters.items():
        # Try to convert to int, float, or bool as appropriate
        try:
            # Convert to int if it looks like an int
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                processed_params[key] = int(value)
            # Convert to float if it has a decimal point
            elif '.' in value:
                try:
                    processed_params[key] = float(value)
                except ValueError:
                    processed_params[key] = value
            # Handle boolean values
            elif value.lower() in ['true', 'false']:
                processed_params[key] = value.lower() == 'true'
            else:
                processed_params[key] = value
        except (AttributeError, ValueError):
            # If conversion fails, keep as string
            processed_params[key] = value

    return processed_params


# Load training data
def load_data():
    train_path = os.path.join(input_path, training_channel_name)

    # Get all CSV files in training directory
    train_files = [os.path.join(train_path, file) for file in os.listdir(train_path)
                   if file.endswith('.csv')]

    if not train_files:
        raise ValueError(f"No CSV files found in {train_path}")

    # Read and concatenate all training files
    dfs = []
    for file in train_files:
        df = pd.read_csv(file)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data found in training files")

    return pd.concat(dfs, ignore_index=True)


# Train the model
def train():
    print("Starting the training process")

    try:
        # Load hyperparameters
        hyperparameters = load_hyperparameters()
        print(f"Loaded hyperparameters: {hyperparameters}")

        # Load training data
        train_data = load_data()
        print(f"Loaded training data with shape: {train_data.shape}")

        # Extract features and target
        # Assumes last column is the target
        X = train_data.iloc[:, :-1]
        y = train_data.iloc[:, -1]

        # Train/test split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Configure model parameters from hyperparameters or use defaults
        max_depth = hyperparameters.get('max_depth', 6)
        learning_rate = hyperparameters.get('learning_rate', 0.1)
        n_estimators = hyperparameters.get('n_estimators', 100)

        # Create and train model with a simpler approach
        # Removed early stopping and eval_set to ensure compatibility
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators
        )

        print("Training model...")
        model.fit(X_train, y_train)

        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        print(f"Validation R² score: {val_score:.4f}")

        # Save the model
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, 'model.joblib')

        # Save additional metadata about the model
        feature_names = X.columns.tolist()
        model_metadata = {
            'feature_names': feature_names,
            'hyperparameters': hyperparameters,
            'validation_score': val_score
        }
        metadata_file = os.path.join(model_path, 'metadata.json')

        print(f"Saving model to {model_file}")
        joblib.dump(model, model_file)

        print(f"Saving metadata to {metadata_file}")
        with open(metadata_file, 'w') as f:
            json.dump(model_metadata, f)

        print("Training completed successfully")

    except Exception as e:
        # Write out an error file
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed
        sys.exit(255)


if __name__ == '__main__':
    train()