"""PyTorch Tabular utilities for Workbench models."""

import os

# Force CPU mode BEFORE any PyTorch imports to avoid MPS/CUDA issues on Mac
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
torch.set_default_device("cpu")
if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = lambda: False

import logging
import tempfile
import numpy as np
import pandas as pd
from typing import Any

# Sklearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
)
from scipy.stats import spearmanr

log = logging.getLogger("workbench")


def expand_proba_column(df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Expands a 'pred_proba' column containing probability lists into separate columns.

    Args:
        df: DataFrame containing a "pred_proba" column
        class_labels: List of class labels

    Returns:
        DataFrame with the "pred_proba" expanded into separate columns
    """
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
    return df


def download_and_extract_model(s3_uri: str, model_dir: str) -> None:
    """Download model artifact from S3 and extract it.

    Args:
        s3_uri: S3 URI to the model artifact (model.tar.gz)
        model_dir: Directory to extract model artifacts to
    """
    import tarfile
    import boto3

    print(f"Downloading model from {s3_uri}...")

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Download to temp file
    s3 = boto3.client("s3")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = tmp.name
        s3.download_file(bucket, key, tmp_path)
        print(f"Downloaded to {tmp_path}")

    # Extract
    print(f"Extracting to {model_dir}...")
    with tarfile.open(tmp_path, "r:gz") as tar:
        tar.extractall(model_dir)

    # Cleanup temp file
    os.unlink(tmp_path)

    # List contents
    print("Model directory contents:")
    for root, dirs, files in os.walk(model_dir):
        level = root.replace(model_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in files:
            print(f"{sub_indent}{file}")


def load_pytorch_model_artifacts(model_dir: str) -> tuple[Any, dict]:
    """Load PyTorch Tabular model and artifacts from an extracted model directory.

    Args:
        model_dir: Directory containing extracted model artifacts

    Returns:
        Tuple of (TabularModel, artifacts_dict).
        artifacts_dict contains 'label_encoder' and 'category_mappings' if present.
    """
    import json
    import joblib

    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    from pytorch_tabular import TabularModel

    # Load the TabularModel
    model_path = os.path.join(model_dir, "tabular_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No tabular_model directory found in {model_dir}")

    # Change to /tmp for PyTorch Tabular (needs write access)
    original_cwd = os.getcwd()
    try:
        os.chdir("/tmp")
        model = TabularModel.load_model(model_path)
    finally:
        os.chdir(original_cwd)

    # Load additional artifacts
    artifacts = {}

    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        artifacts["label_encoder"] = joblib.load(label_encoder_path)

    category_mappings_path = os.path.join(model_dir, "category_mappings.json")
    if os.path.exists(category_mappings_path):
        with open(category_mappings_path) as f:
            artifacts["category_mappings"] = json.load(f)

    return model, artifacts


def cross_fold_inference(
    workbench_model: Any,
    nfolds: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Performs K-fold cross-validation for PyTorch Tabular models.

    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)

    Returns:
        Tuple of:
            - DataFrame with per-class metrics (and 'all' row for overall metrics)
            - DataFrame with columns: id, target, prediction, and *_proba columns (for classifiers)
    """
    from workbench.api import FeatureSet
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    import shutil

    # Create a temporary model directory (caller manages cleanup)
    model_dir = tempfile.mkdtemp(prefix="pytorch_cv_")
    print(f"Using model directory: {model_dir}")

    try:
        # Download and extract model artifacts
        model_artifact_uri = workbench_model.model_data_url()
        download_and_extract_model(model_artifact_uri, model_dir)

        # Load model and artifacts
        loaded_model, artifacts = load_pytorch_model_artifacts(model_dir)
        category_mappings = artifacts.get("category_mappings", {})

        # Extract configs from loaded model
        config = loaded_model.config

        # Determine if classifier
        is_classifier = config.task == "classification"

        # Use saved label encoder if available to maintain class ordering
        if is_classifier:
            label_encoder = artifacts.get("label_encoder")
            if label_encoder is None:
                log.warning("No saved label encoder found, creating fresh one")
                label_encoder = LabelEncoder()
        else:
            label_encoder = None

        # Prepare data
        fs = FeatureSet(workbench_model.get_input())
        df = workbench_model.training_view().pull_dataframe()

        # Get columns
        id_col = fs.id_column
        target_col = workbench_model.target()
        feature_cols = workbench_model.features()
        print(f"Target column: {target_col}")
        print(f"Feature columns: {len(feature_cols)} features")

        # Convert string columns to category for PyTorch Tabular compatibility
        for col in feature_cols:
            if pd.api.types.is_string_dtype(df[col]):
                if col in category_mappings:
                    # Use saved category mappings for consistency
                    df[col] = pd.Categorical(df[col], categories=category_mappings[col])
                else:
                    df[col] = df[col].astype("category")

        # Determine categorical and continuous columns
        categorical_cols = [col for col in feature_cols if df[col].dtype.name == "category"]
        continuous_cols = [col for col in feature_cols if col not in categorical_cols]

        # Cast continuous columns to float
        df[continuous_cols] = df[continuous_cols].astype("float64")

        X = df[feature_cols]
        y = df[target_col]
        ids = df[id_col]

        # Encode target if classifier
        if label_encoder is not None:
            # Fit only if this is a fresh encoder (not loaded from artifacts)
            if not hasattr(label_encoder, 'classes_') or label_encoder.classes_ is None:
                label_encoder.fit(y)
            y_encoded = label_encoder.transform(y)
            y_for_cv = pd.Series(y_encoded, index=y.index, name=target_col)
        else:
            y_for_cv = y

        # Prepare KFold
        kfold = (StratifiedKFold if is_classifier else KFold)(n_splits=nfolds, shuffle=True, random_state=42)

        # Initialize results collection
        fold_metrics = []
        predictions_df = pd.DataFrame({id_col: ids, target_col: y})

        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y_for_cv), 1):
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx}/{nfolds}")
            print(f"{'='*50}")

            # Split data
            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            # Encode target for this fold
            if is_classifier:
                df_train[target_col] = label_encoder.transform(df_train[target_col])
                df_val[target_col] = label_encoder.transform(df_val[target_col])

            # Create fresh configs for this fold
            data_config = DataConfig(
                target=[target_col],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
                num_workers=0,  # Avoid multiprocessing issues on Mac
                pin_memory=False,  # Disable pin_memory to avoid MPS segfault
            )

            trainer_config = TrainerConfig(
                auto_lr_find=config.get("auto_lr_find", False),
                batch_size=config.get("batch_size", 64),
                max_epochs=config.get("max_epochs", 100),
                min_epochs=config.get("min_epochs", 10),
                early_stopping=config.get("early_stopping", "valid_loss"),
                early_stopping_patience=config.get("early_stopping_patience", 10),
                checkpoints=None,  # Disable checkpointing for cross-validation
                accelerator="cpu",  # Force CPU
                devices=1,
                progress_bar="rich",  # Show progress
                gradient_clip_val=config.get("gradient_clip_val", 1.0),
                trainer_kwargs={"enable_model_summary": False},  # Reduce output noise
            )

            optimizer_config = OptimizerConfig()

            # Recreate model config based on type
            # Note: We use CategoryEmbeddingModelConfig as default, could be extended
            from pytorch_tabular.models import CategoryEmbeddingModelConfig

            model_config = CategoryEmbeddingModelConfig(
                task="classification" if is_classifier else "regression",
                layers=config.get("layers", "256-128-64"),
                activation=config.get("activation", "LeakyReLU"),
                learning_rate=config.get("learning_rate", 1e-3),
                dropout=config.get("dropout", 0.3),
                use_batch_norm=config.get("use_batch_norm", True),
                initialization=config.get("initialization", "kaiming"),
            )

            # Create and train fresh model
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )

            # Change to /tmp for training (PyTorch Tabular needs write access)
            original_cwd = os.getcwd()
            try:
                os.chdir("/tmp")
                tabular_model.fit(train=df_train, validation=df_val)
            finally:
                os.chdir(original_cwd)

            # Make predictions
            result = tabular_model.predict(df_val[feature_cols])

            # Extract predictions
            prediction_col = f"{target_col}_prediction"
            preds = result[prediction_col].values

            # Store predictions
            val_indices = df.iloc[val_idx].index
            if is_classifier:
                preds_decoded = label_encoder.inverse_transform(preds.astype(int))
                predictions_df.loc[val_indices, "prediction"] = preds_decoded

                # Get probabilities
                prob_cols = sorted([col for col in result.columns if col.endswith("_probability")])
                if prob_cols:
                    probs = result[prob_cols].values
                    all_proba = pd.Series([None] * len(predictions_df), index=predictions_df.index, dtype=object)
                    all_proba.loc[val_indices] = [p.tolist() for p in probs]
                    predictions_df["pred_proba"] = all_proba
            else:
                predictions_df.loc[val_indices, "prediction"] = preds

            # Calculate fold metrics
            if is_classifier:
                y_val_orig = label_encoder.inverse_transform(df_val[target_col])
                preds_orig = preds_decoded

                # Overall weighted metrics
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average="weighted", zero_division=0
                )

                # Per-class metrics
                prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average=None, zero_division=0, labels=label_encoder.classes_
                )

                # ROC-AUC
                y_val_encoded = df_val[target_col].values
                roc_auc_overall = roc_auc_score(y_val_encoded, probs, multi_class="ovr", average="macro")
                roc_auc_per_class = roc_auc_score(y_val_encoded, probs, multi_class="ovr", average=None)

                fold_metrics.append({
                    "fold": fold_idx,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": roc_auc_overall,
                    "precision_per_class": prec_per_class,
                    "recall_per_class": rec_per_class,
                    "f1_per_class": f1_per_class,
                    "roc_auc_per_class": roc_auc_per_class,
                })

                print(f"Fold {fold_idx} - F1: {f1:.4f}, ROC-AUC: {roc_auc_overall:.4f}")
            else:
                y_val = df_val[target_col].values
                spearman_corr, _ = spearmanr(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))

                fold_metrics.append({
                    "fold": fold_idx,
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_val, preds),
                    "medae": median_absolute_error(y_val, preds),
                    "r2": r2_score(y_val, preds),
                    "spearmanr": spearman_corr,
                })

                print(f"Fold {fold_idx} - RMSE: {rmse:.4f}, R2: {fold_metrics[-1]['r2']:.4f}")

        # Calculate summary metrics
        fold_df = pd.DataFrame(fold_metrics)

        if is_classifier:
            # Expand the pred_proba column
            if "pred_proba" in predictions_df.columns:
                predictions_df = expand_proba_column(predictions_df, label_encoder.classes_)

            # Build per-class metrics DataFrame
            metric_rows = []

            # Per-class rows
            for idx, class_name in enumerate(label_encoder.classes_):
                prec_scores = np.array([fold["precision_per_class"][idx] for fold in fold_metrics])
                rec_scores = np.array([fold["recall_per_class"][idx] for fold in fold_metrics])
                f1_scores = np.array([fold["f1_per_class"][idx] for fold in fold_metrics])
                roc_auc_scores = np.array([fold["roc_auc_per_class"][idx] for fold in fold_metrics])

                y_orig = label_encoder.inverse_transform(y_for_cv)
                support = int((y_orig == class_name).sum())

                metric_rows.append({
                    "class": class_name,
                    "precision": prec_scores.mean(),
                    "recall": rec_scores.mean(),
                    "f1": f1_scores.mean(),
                    "roc_auc": roc_auc_scores.mean(),
                    "support": support,
                })

            # Overall 'all' row
            metric_rows.append({
                "class": "all",
                "precision": fold_df["precision"].mean(),
                "recall": fold_df["recall"].mean(),
                "f1": fold_df["f1"].mean(),
                "roc_auc": fold_df["roc_auc"].mean(),
                "support": len(y_for_cv),
            })

            metrics_df = pd.DataFrame(metric_rows)
        else:
            # Regression metrics
            metrics_df = pd.DataFrame([{
                "rmse": fold_df["rmse"].mean(),
                "mae": fold_df["mae"].mean(),
                "medae": fold_df["medae"].mean(),
                "r2": fold_df["r2"].mean(),
                "spearmanr": fold_df["spearmanr"].mean(),
                "support": len(y_for_cv),
            }])

        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        print(metrics_df.to_string(index=False))

        return metrics_df, predictions_df

    finally:
        # Cleanup model directory
        print(f"\nCleaning up model directory: {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)


def main():
    # Tests for the PyTorch utilities
    from workbench.api import Model, Endpoint

    # Initialize Workbench model
    model_name = "aqsol-pytorch-class"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Perform cross-fold inference
    end = Endpoint(model.endpoints()[0])
    end.cross_fold_inference()


if __name__ == "__main__":
    main()