# Welcome to the custom script template
#
# This script is designed to be used with the SageMaker Scikit-learn container.
# It runs a simple training and inference job using a Scikit-learn model.
# This script will work 'as is' just to get you started, but you should modify it
# to suit your own use case.
#
# We've placed a few comments throughout the script to help guide you.
# Each section with have a 'REPLACE:' in the comment to indicate where you should
# replace the code with your own.
#
# For Regression Models, basically remove all the LabelEncoder code.
#
# Also review the requirements.txt file to ensure you have all the necessary libraries.
#
# If you have any questions, please don't hesitate to touch base with SageWorks support.
#
import argparse
import os
import json
import joblib
from io import StringIO
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# REPLACE: Add any additional imports you need here


# These are helper functions that will should either be used or replaced
# Function to check if the dataframe is empty
def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    if df.empty:
        raise ValueError(f"*** The training data {df_name} has 0 rows! ***STOPPING***")


# Function to expand probability columns for classification
def expand_proba_column(df: pd.DataFrame, class_labels: list) -> pd.DataFrame:
    proba_df = pd.DataFrame(df.pop("pred_proba").tolist(), columns=[f"{label}_proba" for label in class_labels])
    return pd.concat([df.reset_index(drop=True), proba_df], axis=1)


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    args = parser.parse_args()

    # Load the training data
    training_files = [os.path.join(args.train, file) for file in os.listdir(args.train) if file.endswith(".csv")]
    df = pd.concat([pd.read_csv(file) for file in training_files])

    check_dataframe(df, "training_df")

    # Define target and features
    # REPLACE: Update the target and feature list as needed
    target = "wine_class"
    feature_list = [col for col in df.columns if col != target and is_numeric_dtype(df[col])]

    # Encode target labels
    # REPLACE: Label encoding is required for classification problems but not for regression
    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])

    # Split data into training and validation sets
    # REPLACE: Update with your own train/test split logic
    X_train, X_val, y_train, y_val = train_test_split(df[feature_list], df[target], test_size=0.2, random_state=42)

    # Train a simple random forest classifier
    # REPLACE: Update with your Model class and parameters
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_val)

    # Get the probabilities for each class
    # REPLACE: Only for classification models
    probs = model.predict_proba(X_val)

    # Prepare the validation DataFrame with predictions and probabilities
    df_val = X_val.copy()
    df_val["prediction"] = label_encoder.inverse_transform(preds)
    df_val["pred_proba"] = [p.tolist() for p in probs]
    df_val = expand_proba_column(df_val, label_encoder.classes_)

    # Save the model and label encoder
    joblib.dump(model, os.path.join(args.model_dir, "scikit_model.joblib"))
    joblib.dump(label_encoder, os.path.join(args.model_dir, "label_encoder.joblib"))

    # Save the feature list
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(feature_list, fp)

    # Save validation predictions locally to the output directory
    output_path = os.path.join(args.output_data_dir, "validation_predictions.csv")
    df_val.to_csv(output_path, index=False)


# Model loading and prediction functions
def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "scikit_model.joblib"))


# Input and output functions (basically convert CSV to DataFrame)
def input_fn(input_data, content_type):
    if content_type == "text/csv":
        return pd.read_csv(StringIO(input_data))
    raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    if accept_type == "text/csv":
        return output_df.to_csv(index=False), "text/csv"
    raise RuntimeError(f"{accept_type} not supported!")


# Prediction function
def predict_fn(df, model):
    model_dir = os.environ["SM_MODEL_DIR"]
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)

    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    preds = model.predict(df[model_features])
    probs = model.predict_proba(df[model_features])

    df["prediction"] = label_encoder.inverse_transform(preds)
    df["pred_proba"] = [p.tolist() for p in probs]
    return expand_proba_column(df, label_encoder.classes_)
