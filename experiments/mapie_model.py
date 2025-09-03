# Model: LightGBM with MAPIE ConformalizedQuantileRegressor
from mapie.regression import ConformalizedQuantileRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

# Model Performance Scores
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from io import StringIO
import json
import argparse
import joblib
import os
import numpy as np
import pandas as pd
from typing import List, Tuple

# Template Placeholders
TEMPLATE_PARAMS = {
    "target": "solubility",
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
    "train_all_data": False,
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

    # Rename the DataFrame columns to match the model features
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
    """Prepare features for the model by decompressing bitstring features

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
    decompressed_features = features.copy()
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


if __name__ == "__main__":
    # Template Parameters
    target = TEMPLATE_PARAMS["target"]
    features = TEMPLATE_PARAMS["features"]
    orig_features = features.copy()
    compressed_features = TEMPLATE_PARAMS["compressed_features"]
    train_all_data = TEMPLATE_PARAMS["train_all_data"]
    validation_split = 0.2

    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "."))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "."))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
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
        print("WARNING: MAPIE needs Validation data for calibration (required for CQR)")
        print("WARNING: Setting train_all_data does nothing for MAPIE models!")

    # Does the dataframe have a training column?
    if "training" in all_df.columns:
        print("Found training column, splitting data based on training column")
        df_train = all_df[all_df["training"]]
        df_val = all_df[~all_df["training"]]
    else:
        # Just do a random training Split
        print("WARNING: No training column found, splitting data with random state=42")
        df_train, df_val = train_test_split(all_df, test_size=validation_split, random_state=42)
    print(f"FIT/TRAIN: {df_train.shape}")
    print(f"VALIDATION: {df_val.shape}")

    # Prepare features and targets for training
    X_train = df_train[features]
    X_validate = df_val[features]
    y_train = df_train[target]
    y_validate = df_val[target]

    # Train quantile models for CQR - this gives adaptive, asymmetric intervals
    print("Training quantile models for CQR (adaptive intervals)...")
    confidence_level = 0.95  # 95% confidence intervals
    alpha = 1 - confidence_level  # 0.05

    quantile_estimators = []
    quantiles = [alpha / 2, 1 - alpha / 2, 0.5]  # [0.025, 0.975, 0.5] - lower, upper, median

    for q in quantiles:
        print(f"  Training model for quantile {q:.3f}...")
        est = LGBMRegressor(
            objective="quantile",
            alpha=q,  # The quantile to predict
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            force_col_wise=True,  # Better performance with many features
        )
        est.fit(X_train, y_train)
        quantile_estimators.append(est)

    # Create MAPIE CQR model with pre-trained quantile models
    print("Setting up MAPIE ConformalizedQuantileRegressor...")
    model = ConformalizedQuantileRegressor(
        quantile_estimators,
        confidence_level=confidence_level,  # Single confidence level for v1.0+
        prefit=True,  # Models are already trained
    )

    # Conformalize the model using validation set
    # This calibrates the intervals to achieve the desired coverage
    print("Conformalizing CQR model with validation data...")
    model.conformalize(X_validate, y_validate)

    # Make Predictions on the Validation Set using MAPIE
    print(f"Making Predictions on Validation Set...")
    y_pred_mapie, y_pis_mapie = model.predict_interval(X_validate)

    # Calculate various model performance metrics (regression)
    rmse = root_mean_squared_error(y_validate, y_pred_mapie)
    mae = mean_absolute_error(y_validate, y_pred_mapie)
    r2 = r2_score(y_validate, y_pred_mapie)
    print(f"\nPoint Prediction Performance:")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"NumRows: {len(df_val)}")

    # Calculate empirical coverage
    print("\nInterval Coverage:")
    coverage = np.mean((y_validate >= y_pis_mapie[:, 0, 0]) & (y_validate <= y_pis_mapie[:, 1, 0]))
    print(f"  Target: {confidence_level * 100:.0f}%, Empirical: {coverage * 100:.1f}%")

    # Calculate interval statistics - CQR should show adaptive intervals
    interval_widths = y_pis_mapie[:, 1, 0] - y_pis_mapie[:, 0, 0]
    print(f"\nInterval Statistics (Adaptive):")
    print(f"  Average width: {np.mean(interval_widths):.3f}")
    print(f"  Median width: {np.median(interval_widths):.3f}")
    print(f"  Width std: {np.std(interval_widths):.3f}")
    print(f"  Min width: {np.min(interval_widths):.3f}")
    print(f"  Max width: {np.max(interval_widths):.3f}")

    # Check for asymmetry in intervals
    lower_dists = y_pred_mapie - y_pis_mapie[:, 0, 0]
    upper_dists = y_pis_mapie[:, 1, 0] - y_pred_mapie
    asymmetry = np.mean(upper_dists - lower_dists)
    print(f"\nInterval Asymmetry:")
    print(f"  Mean asymmetry: {asymmetry:.3f} (positive = upper-skewed)")
    print(f"  % asymmetric: {np.mean(np.abs(upper_dists - lower_dists) > 0.01) * 100:.1f}%")

    # Save the trained MAPIE model
    joblib.dump(model, os.path.join(args.model_dir, "mapie_model.joblib"))

    # Save the feature list to validate input during predictions
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as fp:
        json.dump(features, fp)

    # Save category mappings if any
    if category_mappings:
        with open(os.path.join(args.model_dir, "category_mappings.json"), "w") as fp:
            json.dump(category_mappings, fp)

    # Save model configuration for reference
    model_config = {
        "model_type": "MAPIE_CQR_LightGBM",
        "confidence_level": confidence_level,
        "n_features": len(features),
        "target": target,
        "validation_metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "coverage": float(coverage),
            "avg_interval_width": float(np.mean(interval_widths)),
            "interval_width_std": float(np.std(interval_widths)),
        },
    }
    with open(os.path.join(args.model_dir, "model_config.json"), "w") as fp:
        json.dump(model_config, fp, indent=2)

    print(f"\nModel training complete! Saved to {args.model_dir}")


#
# Inference Section
#
def model_fn(model_dir) -> dict:
    """Load and return the MAPIE model from the specified directory."""

    # Load MAPIE Model
    mapie_model = joblib.load(os.path.join(model_dir, "mapie_model.joblib"))

    # Load category mappings if they exist
    category_mappings = {}
    category_path = os.path.join(model_dir, "category_mappings.json")
    if os.path.exists(category_path):
        with open(category_path) as fp:
            category_mappings = json.load(fp)

    return {"mapie": mapie_model, "category_mappings": category_mappings}


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
        # Convert categorical columns to string to avoid fillna issues
        for col in output_df.select_dtypes(include=["category"]).columns:
            output_df[col] = output_df[col].astype(str)
        csv_output = output_df.fillna("N/A").to_csv(index=False)  # CSV with N/A for missing values
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


def predict_fn(df, models) -> pd.DataFrame:
    """Make Predictions with MAPIE CQR - provides adaptive, asymmetric intervals

    Args:
        df (pd.DataFrame): The input DataFrame
        models (dict): The dictionary containing the MAPIE model

    Returns:
        pd.DataFrame: The DataFrame with predictions and adaptive uncertainty intervals
    """

    # Grab our feature columns (from training)
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    with open(os.path.join(model_dir, "feature_columns.json")) as fp:
        model_features = json.load(fp)

    # Match features in a case-insensitive manner
    matched_df = match_features_case_insensitive(df, model_features)

    # Apply categorical mappings if they exist
    if models.get("category_mappings"):
        matched_df, _ = convert_categorical_types(matched_df, model_features, models["category_mappings"])

    # Get CQR predictions with adaptive 95% confidence intervals
    y_pred, y_pis = models["mapie"].predict_interval(matched_df[model_features])

    # Primary outputs - CQR provides adaptive, asymmetric 95% intervals
    df["prediction"] = y_pred  # Median prediction from quantile model
    df["q_025"] = y_pis[:, 0, 0]  # 95% interval lower bound
    df["q_975"] = y_pis[:, 1, 0]  # 95% interval upper bound

    # Calculate uncertainty metrics
    interval_width = df["q_975"] - df["q_025"]
    df["prediction_std"] = interval_width / 3.92  # Approximate std from 95% interval

    # Calculate asymmetry for preserving in approximations
    lower_dist = df["prediction"] - df["q_025"]
    upper_dist = df["q_975"] - df["prediction"]
    df["interval_asymmetry"] = (upper_dist - lower_dist) / interval_width

    # Approximate other quantiles by scaling from 95% interval
    # Direct mapping to avoid floating point issues
    quantile_approximations = [
        ("q_05", "q_95", 0.84),  # 90% confidence
        ("q_10", "q_90", 0.68),  # 80% confidence
        ("q_25", "q_75", 0.37),  # 50% confidence
    ]

    for lower_name, upper_name, scale in quantile_approximations:
        # Scale each side independently to preserve asymmetry
        df[lower_name] = df["prediction"] - scale * lower_dist
        df[upper_name] = df["prediction"] + scale * upper_dist

    # Uncertainty metrics
    df["uncertainty_score"] = interval_width / (np.abs(df["prediction"]) + 1e-6)

    # Flag high uncertainty predictions
    uncertainty_threshold = df["uncertainty_score"].quantile(0.9)
    df["high_uncertainty"] = df["uncertainty_score"] > uncertainty_threshold

    # Confidence bands for decision stages
    df["confidence_band"] = pd.cut(
        df["uncertainty_score"], bins=[0, 0.5, 1.0, 2.0, np.inf], labels=["high", "medium", "low", "very_low"]
    )
    return df
