# Imports for PyTorch Tabular Model
import os
import awswrangler as wr
import numpy as np

# PyTorch compatibility: pytorch-tabular saves complex objects, not just tensors
# Use legacy loading behavior for compatibility (recommended by PyTorch docs for this scenario)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig

# Model Performance Scores
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    precision_recall_fscore_support,
    confusion_matrix,
)

# Classification Encoder
from sklearn.preprocessing import LabelEncoder

# Scikit Learn Imports
from sklearn.model_selection import train_test_split

from io import StringIO
import json
import argparse
import joblib
import os
import pandas as pd
from typing import List, Tuple

# Template Parameters
TEMPLATE_PARAMS = {
    # "model_type": "regressor",
    "model_type": "classifier",
    # "target_column": "solubility",
    "target_column": "solubility_class",
    "features": [
        "molwt",
        "mollogp",
        "molmr",
        "heavyatomcount",
        "numhacceptors",
        "numhdonors",
        "numheteroatoms",
        "numrotatablebonds",
        "numvalenceelectrons",
        "numaromaticrings",
        "numsaturatedrings",
        "numaliphaticrings",
        "ringcount",
        "tpsa",
        "labuteasa",
        "balabanj",
        "bertzct",
    ],
    "compressed_features": [],
    "model_metrics_s3_path": "s3://sandbox-sageworks-artifacts/models/aqsol-pytorch-reg/training",
    "train_all_data": False,
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "layers": "512-256",
        "activation": "ReLU",
        "dropout": 0.1,
        "use_batch_norm": True,
        "initialization": "kaiming",
    },
}


# Function to check if dataframe is empty
def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    """
    Check if the provided dataframe is empty and raise an exception if it is.

    Args:
        df (pd.DataFrame): DataFrame to check
        df_name (str): Name of the DataFrame
    """
    if df.empty:
        msg = f"*** The training data {df_name} has 0 rows! ***STOPPING***"
        print(msg)
        raise ValueError(msg)


def expand_proba_column(df: pd.DataFrame, class_labels: List[str]) -> pd.DataFrame:
    """
    Expands a column in a DataFrame containing a list of probabilities into separate columns.

    Args:
        df (pd.DataFrame): DataFrame containing a "pred_proba" column
        class_labels (List[str]): List of class labels

    Returns:
        pd.DataFrame: DataFrame with the "pred_proba" expanded into separate columns
    """

    # Sanity check
    proba_column = "pred_proba"
    if proba_column not in df.columns:
        raise ValueError('DataFrame does not contain a "pred_proba" column')

    # Construct new column names with '_proba' suffix
    proba_splits = [f"{label}_proba" for label in class_labels]

    # Expand the proba_column into separate columns for each probability
    proba_df = pd.DataFrame(df[proba_column].tolist(), columns=proba_splits)

    # Drop any proba columns and reset the index in prep for the concat
    df = df.drop(columns=[proba_column] + proba_splits, errors="ignore")
    df = df.reset_index(drop=True)

    # Concatenate the new columns with the original DataFrame
    df = pd.concat([df, proba_df], axis=1)
    print(df)
    return df


def match_features_case_insensitive(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """
    Matches and renames DataFrame columns to match model feature names (case-insensitive).
    Prioritizes exact matches, then case-insensitive matches.

    Raises ValueError if any model features cannot be matched.
    """
    df_columns_lower = {col.lower(): col for col in df.columns}
    rename_dict = {}
    missing = []

    for feature in model_features:
        if feature in df.columns:
            continue  # Exact match
        elif feature.lower() in df_columns_lower:
            rename_dict[df_columns_lower[feature.lower()]] = feature
        else:
            missing.append(feature)

    if missing:
        raise ValueError(f"Features not found: {missing}")

    return df.rename(columns=rename_dict)


def convert_categorical_types(df: pd.DataFrame, features: list, category_mappings={}) -> tuple:
    """
    Converts appropriate columns to categorical type with consistent mappings.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        features (list): List of feature names to consider for conversion.
        category_mappings (dict, optional): Existing category mappings. If empty dict, we're in
                                            training mode. If populated, we're in inference mode.

    Returns:
        tuple: (processed DataFrame, category mappings dictionary)
    """
    # Training mode
    if category_mappings == {}:
        for col in df.select_dtypes(include=["object", "string"]):
            if col in features and df[col].nunique() < 20:
                print(f"Training mode: Converting {col} to category")
                df[col] = df[col].astype("category")
                category_mappings[col] = df[col].cat.categories.tolist()  # Store category mappings

    # Inference mode
    else:
        for col, categories in category_mappings.items():
            if col in df.columns:
                print(f"Inference mode: Applying categorical mapping for {col}")
                df[col] = pd.Categorical(df[col], categories=categories)  # Apply consistent categorical mapping

    return df, category_mappings


def decompress_features(
    df: pd.DataFrame, features: List[str], compressed_features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for the model

    Args:
        df (pd.DataFrame): The features DataFrame
        features (List[str]): Full list of feature names
        compressed_features (List[str]): List of feature names to decompress (bitstrings)

    Returns:
        pd.DataFrame: DataFrame with the decompressed features
        List[str]: Updated list of feature names after decompression

    Raises:
        ValueError: If any missing values are found in the specified features
    """

    # Check for any missing values in the required features
    missing_counts = df[features].isna().sum()
    if missing_counts.any():
        missing_features = missing_counts[missing_counts > 0]
        print(
            f"WARNING: Found missing values in features: {missing_features.to_dict()}. "
            "WARNING: You might want to remove/replace all NaN values before processing."
        )

    # Decompress the specified compressed features
    decompressed_features = features
    for feature in compressed_features:
        if (feature not in df.columns) or (feature not in features):
            print(f"Feature '{feature}' not in the features list, skipping decompression.")
            continue

        # Remove the feature from the list of features to avoid duplication
        decompressed_features.remove(feature)

        # Handle all compressed features as bitstrings
        bit_matrix = np.array([list(bitstring) for bitstring in df[feature]], dtype=np.uint8)
        prefix = feature[:3]

        # Create all new columns at once - avoids fragmentation
        new_col_names = [f"{prefix}_{i}" for i in range(bit_matrix.shape[1])]
        new_df = pd.DataFrame(bit_matrix, columns=new_col_names, index=df.index)

        # Add to features list
        decompressed_features.extend(new_col_names)

        # Drop original column and concatenate new ones
        df = df.drop(columns=[feature])
        df = pd.concat([df, new_df], axis=1)

    return df, decompressed_features


def model_fn(model_dir):
    """Deserialize and return fitted PyTorch Tabular model"""
    model_path = os.path.join(model_dir, "tabular_model")
    model = TabularModel.load_model(model_path)
    return model


def input_fn(input_data, content_type):
    """Parse input data and return a DataFrame."""
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    # Decode bytes to string if necessary
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))  # Assumes JSON array of records
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    """Supports both CSV and JSON output formats."""
    if "text/csv" in accept_type:
        csv_output = output_df.fillna("N/A").to_csv(index=False)  # CSV with N/A for missing values
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


def predict_fn(df, model) -> pd.DataFrame:
    """Make Predictions with our PyTorch Tabular Model

    Args:
        df (pd.DataFrame): The input DataFrame
        model: The TabularModel use for predictions

    Returns:
        pd.DataFrame: The DataFrame with the predictions added
    """
    compressed_features = TEMPLATE_PARAMS["compressed_features"]

    # Grab our feature columns (from training)
    model_dir = os.environ.get("SM_MODEL_DIR", "pytorch_outputs")
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        features = json.load(fp)
    print(f"Model Features: {features}")

    # Load the category mappings (from training)
    with open(os.path.join(model_dir, "category_mappings.json")) as fp:
        category_mappings = json.load(fp)

    # Load our Label Encoder if we have one
    label_encoder = None
    if os.path.exists(os.path.join(model_dir, "label_encoder.joblib")):
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    # We're going match features in a case-insensitive manner, accounting for all the permutations
    # - Model has a feature list that's any case ("Id", "taCos", "cOunT", "likes_tacos")
    # - Incoming data has columns that are mixed case ("ID", "Tacos", "Count", "Likes_Tacos")
    matched_df = match_features_case_insensitive(df, features)

    # Detect categorical types in the incoming DataFrame
    matched_df, _ = convert_categorical_types(matched_df, features, category_mappings)

    # If we have compressed features, decompress them
    if compressed_features:
        print("Decompressing features for prediction...")
        matched_df, features = decompress_features(matched_df, features, compressed_features)

    # Make predictions using the TabularModel
    result = model.predict(matched_df[features])

    # pytorch-tabular returns predictions using f"{target}_prediction" column
    # and classification probabilities in columns ending with "_probability"
    target = TEMPLATE_PARAMS["target_column"]
    prediction_column = f"{target}_prediction"
    if prediction_column in result.columns:
        predictions = result[prediction_column].values
    else:
        raise ValueError(f"Cannot find prediction column in: {result.columns.tolist()}")

    # If we have a label encoder, decode the predictions
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions.astype(int))

    # Set the predictions on the DataFrame
    df["prediction"] = predictions

    # For classification, get probabilities
    if label_encoder is not None:
        prob_cols = [col for col in result.columns if col.endswith("_probability")]
        if prob_cols:
            probs = result[prob_cols].values
            df["pred_proba"] = [p.tolist() for p in probs]

            # Expand the pred_proba column into separate columns for each class
            df = expand_proba_column(df, label_encoder.classes_)

    # All done, return the DataFrame with new columns for the predictions
    return df


if __name__ == "__main__":
    """The main function is for training the PyTorch Tabular model"""

    # Harness Template Parameters
    target = TEMPLATE_PARAMS["target_column"]
    features = TEMPLATE_PARAMS["features"]
    orig_features = features.copy()
    compressed_features = TEMPLATE_PARAMS["compressed_features"]
    model_type = TEMPLATE_PARAMS["model_type"]
    model_metrics_s3_path = TEMPLATE_PARAMS["model_metrics_s3_path"]
    train_all_data = TEMPLATE_PARAMS["train_all_data"]
    hyperparameters = TEMPLATE_PARAMS["hyperparameters"]
    validation_split = 0.2

    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "pytorch_outputs"))
    args = parser.parse_args()

    # Pull training data from a FeatureSet
    from workbench.api import FeatureSet

    fs = FeatureSet("aqsol_features")
    all_df = fs.pull_dataframe()

    # Check if the dataframe is empty
    check_dataframe(all_df, "training_df")

    # Features/Target output
    print(f"Target: {target}")
    print(f"Features: {str(features)}")

    # Convert any features that might be categorical to 'category' type
    all_df, category_mappings = convert_categorical_types(all_df, features)

    # If we have compressed features, decompress them
    if compressed_features:
        print(f"Decompressing features {compressed_features}...")
        all_df, features = decompress_features(all_df, features, compressed_features)

    # Do we want to train on all the data?
    if train_all_data:
        print("Training on ALL of the data")
        df_train = all_df.copy()
        df_val = all_df.copy()

    # Does the dataframe have a training column?
    elif "training" in all_df.columns:
        print("Found training column, splitting data based on training column")
        df_train = all_df[all_df["training"]]
        df_val = all_df[~all_df["training"]]
    else:
        # Just do a random training Split
        print("WARNING: No training column found, splitting data with random state=42")
        df_train, df_val = train_test_split(all_df, test_size=validation_split, random_state=42)
    print(f"FIT/TRAIN: {df_train.shape}")
    print(f"VALIDATION: {df_val.shape}")

    # Determine categorical and continuous columns
    categorical_cols = [col for col in features if df_train[col].dtype.name == "category"]
    continuous_cols = [col for col in features if col not in categorical_cols]

    print(f"Categorical columns: {categorical_cols}")
    print(f"Continuous columns: {continuous_cols}")

    # Set up PyTorch Tabular configuration
    data_config = DataConfig(
        target=[target],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )

    # Choose the 'task' based on model type also set up the label encoder if needed
    if model_type == "classifier":
        task = "classification"
        # Encode the target column
        label_encoder = LabelEncoder()
        df_train[target] = label_encoder.fit_transform(df_train[target])
        df_val[target] = label_encoder.transform(df_val[target])
    else:
        task = "regression"
        label_encoder = None

    # Use any hyperparameters to set up both the trainer and model configurations
    print(f"Hyperparameters: {hyperparameters}")

    # Set up PyTorch Tabular configuration with defaults
    trainer_defaults = {
        "auto_lr_find": True,
        "batch_size": min(1024, max(32, len(df_train) // 4)),
        "max_epochs": 100,
        "early_stopping": "valid_loss",
        "early_stopping_patience": 15,
        "checkpoints": "valid_loss",
        "accelerator": "auto",
        "progress_bar": "none",
        "gradient_clip_val": 1.0,
    }

    # Override defaults with any provided hyperparameters for trainer
    trainer_params = {**trainer_defaults, **{k: v for k, v in hyperparameters.items() if k in trainer_defaults}}
    trainer_config = TrainerConfig(**trainer_params)

    # Model config defaults
    model_defaults = {
        "layers": "1024-512-512",
        "activation": "ReLU",
        "learning_rate": 1e-3,
        "dropout": 0.1,
        "use_batch_norm": True,
        "initialization": "kaiming",
    }
    # Override defaults with any provided hyperparameters for model
    model_params = {**model_defaults, **{k: v for k, v in hyperparameters.items() if k in model_defaults}}
    # Use CategoryEmbedding for both regression and classification tasks
    model_config = CategoryEmbeddingModelConfig(task=task, **model_params)
    optimizer_config = OptimizerConfig()

    #####################################
    # Create and train the TabularModel #
    #####################################
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=df_train, validation=df_val)

    # Make Predictions on the Validation Set
    print(f"Making Predictions on Validation Set...")
    result = tabular_model.predict(df_val, include_input_features=False)

    # pytorch-tabular returns predictions using f"{target}_prediction" column
    # and classification probabilities in columns ending with "_probability"
    if model_type == "classifier":
        preds = result[f"{target}_prediction"].values
    else:
        # Regression: use the target column name
        preds = result[f"{target}_prediction"].values

    if model_type == "classifier":
        # Get probabilities for classification
        print("Processing Probabilities...")
        prob_cols = [col for col in result.columns if col.endswith("_probability")]
        if prob_cols:
            probs = result[prob_cols].values
            df_val["pred_proba"] = [p.tolist() for p in probs]

            # Expand the pred_proba column into separate columns for each class
            print(df_val.columns)
            df_val = expand_proba_column(df_val, label_encoder.classes_)
            print(df_val.columns)

        # Decode the target and prediction labels
        y_validate = label_encoder.inverse_transform(df_val[target])
        preds = label_encoder.inverse_transform(preds.astype(int))
    else:
        y_validate = df_val[target].values

    # Save predictions to S3 (just the target, prediction, and '_probability' columns)
    df_val["prediction"] = preds
    output_columns = [target, "prediction"]
    output_columns += [col for col in df_val.columns if col.endswith("_probability")]
    wr.s3.to_csv(
        df_val[output_columns],
        path=f"{model_metrics_s3_path}/validation_predictions.csv",
        index=False,
    )

    # Report Performance Metrics
    if model_type == "classifier":
        # Get the label names and their integer mapping
        label_names = label_encoder.classes_

        # Calculate various model performance metrics
        scores = precision_recall_fscore_support(y_validate, preds, average=None, labels=label_names)

        # Put the scores into a dataframe
        score_df = pd.DataFrame(
            {
                target: label_names,
                "precision": scores[0],
                "recall": scores[1],
                "fscore": scores[2],
                "support": scores[3],
            }
        )

        # We need to get creative with the Classification Metrics
        metrics = ["precision", "recall", "fscore", "support"]
        for t in label_names:
            for m in metrics:
                value = score_df.loc[score_df[target] == t, m].iloc[0]
                print(f"Metrics:{t}:{m} {value}")

        # Compute and output the confusion matrix
        conf_mtx = confusion_matrix(y_validate, preds, labels=label_names)
        for i, row_name in enumerate(label_names):
            for j, col_name in enumerate(label_names):
                value = conf_mtx[i, j]
                print(f"ConfusionMatrix:{row_name}:{col_name} {value}")

    else:
        # Calculate various model performance metrics (regression)
        rmse = root_mean_squared_error(y_validate, preds)
        mae = mean_absolute_error(y_validate, preds)
        r2 = r2_score(y_validate, preds)
        print(f"RMSE: {rmse:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"R2: {r2:.3f}")
        print(f"NumRows: {len(df_val)}")

    # Save the model to the standard place/name
    tabular_model.save_model(os.path.join(args.model_dir, "tabular_model"))
    if label_encoder:
        joblib.dump(label_encoder, os.path.join(args.model_dir, "label_encoder.joblib"))

    # Save the features (this will validate input during predictions)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(orig_features, fp)  # We save the original features, not the decompressed ones

    # Save the category mappings
    with open(os.path.join(args.model_dir, "category_mappings.json"), "w") as fp:
        json.dump(category_mappings, fp)

    # Now test the prediction function
    model = model_fn(args.model_dir)
    test_df = df_val
    print(f"Testing model with {len(test_df)} rows...")
    predictions_df = predict_fn(test_df, model)
    print(predictions_df.head())

    # Remove the pytorch_outputs directory if it exists
    if os.path.exists("pytorch_outputs"):
        import shutil

        shutil.rmtree("pytorch_outputs")
        print("Removed pytorch_outputs directory.")
