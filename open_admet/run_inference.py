import numpy as np
import pandas as pd
from workbench.api import Endpoint, DFStore

df_store = DFStore()

# Inverse transform config: how to convert log-space predictions back to original scale
# For log-transformed targets: original = (10^prediction / multiplier) - 1
INVERSE_TRANSFORM_CONFIG = {
    "logd": {"log_transform": False, "multiplier": 1.0},
    "ksol": {"log_transform": True, "multiplier": 1e-6},
    "hlm_clint": {"log_transform": True, "multiplier": 1.0},
    "mlm_clint": {"log_transform": True, "multiplier": 1.0},
    "caco_2_papp_a_b": {"log_transform": True, "multiplier": 1e-6},
    "caco_2_efflux": {"log_transform": True, "multiplier": 1.0},
    "mppb": {"log_transform": True, "multiplier": 1.0},
    "mbpb": {"log_transform": True, "multiplier": 1.0},
    "mgmb": {"log_transform": True, "multiplier": 1.0},
}


def inverse_transform(predictions: np.ndarray, target_name: str) -> np.ndarray:
    """Convert log-space predictions back to original scale."""
    config = INVERSE_TRANSFORM_CONFIG.get(target_name)
    if config is None or not config["log_transform"]:
        return predictions
    # Inverse of: log10((value + 1) * multiplier)
    # => value = (10^prediction / multiplier) - 1
    return (np.power(10, predictions) / config["multiplier"]) - 1


# Mapping from our internal names to submission column names
COLUMN_MAP = {
    "caco_2_efflux": "Caco-2 Permeability Efflux",
    "caco_2_papp_a_b": "Caco-2 Permeability Papp A>B",
    "hlm_clint": "HLM CLint",
    "ksol": "KSOL",
    "logd": "LogD",
    "mbpb": "MBPB",
    "mgmb": "MGMB",
    "mlm_clint": "MLM CLint",
    "mppb": "MPPB",
}

# Map model name prefix to our internal target name
MODEL_TO_TARGET = {
    "caco-2-efflux": "caco_2_efflux",
    "caco-2-papp-a-b": "caco_2_papp_a_b",
    "hlm-clint": "hlm_clint",
    "ksol": "ksol",
    "logd": "logd",
    "mbpb": "mbpb",
    "mgmb": "mgmb",
    "mlm-clint": "mlm_clint",
    "mppb": "mppb",
}

# List of all available models
model_list = [
    "caco-2-efflux-reg-chemprop",
    "caco-2-efflux-reg-pytorch",
    "caco-2-efflux-reg-xgb",
    "caco-2-papp-a-b-reg-chemprop",
    "caco-2-papp-a-b-reg-pytorch",
    "caco-2-papp-a-b-reg-xgb",
    "hlm-clint-reg-chemprop",
    "hlm-clint-reg-pytorch",
    "hlm-clint-reg-xgb",
    "ksol-reg-chemprop",
    "ksol-reg-pytorch",
    "ksol-reg-xgb",
    "logd-reg-chemprop",
    "logd-reg-pytorch",
    "logd-reg-xgb",
    "mbpb-reg-chemprop",
    "mbpb-reg-pytorch",
    "mbpb-reg-xgb",
    "mgmb-reg-chemprop",
    "mgmb-reg-pytorch",
    "mgmb-reg-xgb",
    "mlm-clint-reg-chemprop",
    "mlm-clint-reg-pytorch",
    "mlm-clint-reg-xgb",
    "mppb-reg-chemprop",
    "mppb-reg-pytorch",
    "mppb-reg-xgb",
]

xgb_models = [name for name in model_list if name.endswith("-xgb")]
pytorch_models = [name for name in model_list if name.endswith("-pytorch")]
chemprop_models = [name for name in model_list if name.endswith("-chemprop")]

# Grab test data
test_df = pd.read_csv("test_data_blind.csv")

# Hit Feature Endpoint
"""
rdkit_end = Endpoint("smiles-to-taut-md-stereo-v1")
df_features = rdkit_end.inference(test_df)

# Shove this into the DFStore for faster use later
df_store.upsert("/workbench/datasets/open_admet_test_featurized", df_features)
"""

# Grab featurized test data from DFStore
df_features = df_store.get("/workbench/datasets/open_admet_test_featurized")


def get_model_prefix(model_name: str) -> str:
    """Extract the target prefix from model name (e.g., 'caco-2-efflux-reg-xgb' -> 'caco-2-efflux')"""
    # Remove the model type suffix
    for suffix in ["-reg-chemprop-hybrid", "-reg-chemprop", "-reg-pytorch", "-reg-xgb"]:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def get_predictions_for_model(model_name: str, regenerate: bool = True) -> tuple[str, np.ndarray, np.ndarray]:
    """Run inference for a single model and return (target_name, predictions, prediction_std)."""

    # Check if results already exist in DFStore
    store_path = f"/workbench/open_admet/inference_runs/{model_name}"
    if not regenerate and df_store.get(store_path) is not None:
        print(f"Loading cached inference results for: {model_name}")
        result_df = df_store.get(store_path)
    else:
        print(f"Running inference for: {model_name}")
        end = Endpoint(model_name)
        result_df = end.inference(df_features)
        df_store.upsert(store_path, result_df)

    # Get the target column name from model name
    model_prefix = get_model_prefix(model_name)
    internal_target = MODEL_TO_TARGET[model_prefix]

    # Return predictions and std in log-space
    predictions = result_df["prediction"].values
    prediction_std = result_df["prediction_std"].values

    return internal_target, predictions, prediction_std


def run_inference_and_create_submission(models: list[str], output_file: str):
    """Run inference on a set of models and create a submission CSV."""

    # Start with molecule name and smiles from test data
    submission_df = df_features[["Molecule_Name", "SMILES"]].copy()
    submission_df = submission_df.rename(columns={"Molecule_Name": "Molecule Name"})

    # Run inference for each model and collect predictions
    for model_name in models:
        internal_target, predictions, _ = get_predictions_for_model(model_name)
        submission_col = COLUMN_MAP[internal_target]

        # Apply inverse transform
        original_scale_predictions = inverse_transform(predictions, internal_target)

        submission_df[submission_col] = original_scale_predictions
        config = INVERSE_TRANSFORM_CONFIG.get(internal_target, {})
        print(
            f"  Mapped {internal_target} -> {submission_col} (inverse transform: {config.get('log_transform', False)})"
        )

    # Reorder columns to match submission format
    submission_cols = [
        "Molecule Name",
        "SMILES",
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]
    submission_df = submission_df[submission_cols]

    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"\nSubmission file saved to: {output_file}")
    return submission_df


def run_meta_model_inference(output_file: str = "submission_meta.csv"):
    """Run inference using all 5 model types with inverse-variance weighted averaging.

    For each endpoint, we:
    1. Get predictions and prediction_std from XGBoost, PyTorch, ChemProp, ChemProp Hybrid, and Multi-Target models
    2. Weight each model's prediction by 1/variance (inverse-variance weighting)
    3. Compute weighted average in log-space (before inverse transform)
    4. Apply inverse transform to the weighted prediction
    """
    print("=" * 60)
    print("Creating META-MODEL submission (inverse-variance weighted ensemble)")
    print("=" * 60)

    # Start with molecule name and smiles from test data
    submission_df = df_features[["Molecule_Name", "SMILES"]].copy()
    submission_df = submission_df.rename(columns={"Molecule_Name": "Molecule Name"})

    # Process each target endpoint
    for target_prefix, internal_target in MODEL_TO_TARGET.items():
        print(f"\nProcessing target: {internal_target}")

        # Build model names for the 3 single-target model types
        model_names = [
            f"{target_prefix}-reg-xgb",
            f"{target_prefix}-reg-pytorch",
            f"{target_prefix}-reg-chemprop",
        ]

        # Collect predictions and stds from single-target models (in log-space)
        preds_df = pd.DataFrame()
        stds_df = pd.DataFrame()
        for model_name in model_names:
            _, predictions, prediction_std = get_predictions_for_model(model_name)
            preds_df[model_name] = predictions
            stds_df[model_name] = prediction_std

        # Inverse-variance weighting: weight = 1 / variance = 1 / std^2
        variances = stds_df**2 + 1e-8
        weights = 1.0 / variances
        weights_normalized = weights.div(weights.sum(axis=1), axis=0)

        # Weighted average: sum(weight * prediction) per row
        weighted_predictions = (weights_normalized * preds_df).sum(axis=1)
        print(f"  Inverse-variance weighted predictions from {len(preds_df.columns)} models")

        # Apply inverse transform to the weighted prediction
        original_scale_predictions = inverse_transform(weighted_predictions, internal_target)

        submission_col = COLUMN_MAP[internal_target]
        submission_df[submission_col] = original_scale_predictions
        config = INVERSE_TRANSFORM_CONFIG.get(internal_target, {})
        print(
            f"  Mapped {internal_target} -> {submission_col} (inverse transform: {config.get('log_transform', False)})"
        )

    # Reorder columns to match submission format
    submission_cols = [
        "Molecule Name",
        "SMILES",
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ]
    submission_df = submission_df[submission_cols]

    # Save to CSV
    submission_df.to_csv(output_file, index=False)
    print(f"\nMeta-model submission file saved to: {output_file}")
    return submission_df


if __name__ == "__main__":
    # Create META-MODEL submission (ensemble of all 3 model types)
    # run_meta_model_inference("submission_meta.csv")

    # Optionally create individual model type submissions
    # print("\n" + "=" * 60)
    # print("Creating submission with XGBoost models")
    # print("=" * 60)
    # run_inference_and_create_submission(xgb_models, "submission_xgb.csv")

    # print("\n" + "=" * 60)
    # print("Creating submission with PyTorch models")
    # print("=" * 60)
    # run_inference_and_create_submission(pytorch_models, "submission_pytorch.csv")

    # print("\n" + "=" * 60)
    # print("Creating submission with ChemProp models")
    # print("=" * 60)
    run_inference_and_create_submission(chemprop_models, "submission_chemprop.csv")
