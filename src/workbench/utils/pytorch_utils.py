"""PyTorch Tabular utilities for Workbench models."""

# flake8: noqa: E402
import logging
import os
import tempfile
from pprint import pformat
from typing import Any, Tuple

# Disable OpenMP parallelism to avoid segfaults on macOS with conflicting OpenMP runtimes
# (libomp from LLVM vs libiomp from Intel). Must be set before importing numpy/sklearn/torch.
# See: https://github.com/scikit-learn/scikit-learn/issues/21302
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from workbench.utils.model_utils import safe_extract_tarfile
from workbench.utils.pandas_utils import expand_proba_column
from workbench.utils.aws_utils import pull_s3_data

log = logging.getLogger("workbench")


def download_and_extract_model(s3_uri: str, model_dir: str) -> None:
    """Download model artifact from S3 and extract it.

    Args:
        s3_uri: S3 URI to the model artifact (model.tar.gz)
        model_dir: Directory to extract model artifacts to
    """
    import awswrangler as wr

    log.info(f"Downloading model from {s3_uri}...")

    # Download to temp file
    local_tar_path = os.path.join(model_dir, "model.tar.gz")
    wr.s3.download(path=s3_uri, local_file=local_tar_path)

    # Extract using safe extraction
    log.info(f"Extracting to {model_dir}...")
    safe_extract_tarfile(local_tar_path, model_dir)

    # Cleanup tar file
    os.unlink(local_tar_path)


def load_pytorch_model_artifacts(model_dir: str) -> Tuple[Any, dict]:
    """Load PyTorch Tabular model and artifacts from an extracted model directory.

    Args:
        model_dir: Directory containing extracted model artifacts

    Returns:
        Tuple of (TabularModel, artifacts_dict).
        artifacts_dict contains 'label_encoder' and 'category_mappings' if present.
    """
    import json

    import joblib

    # pytorch-tabular saves complex objects, use legacy loading behavior
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    from pytorch_tabular import TabularModel

    model_path = os.path.join(model_dir, "tabular_model")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No tabular_model directory found in {model_dir}")

    # PyTorch Tabular needs write access, so chdir to /tmp
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


def _extract_model_configs(loaded_model: Any, n_train: int) -> dict:
    """Extract trainer and model configs from a loaded PyTorch Tabular model.

    Args:
        loaded_model: Loaded TabularModel instance
        n_train: Number of training samples (used for batch_size calculation)

    Returns:
        Dictionary with 'trainer' and 'model' config dictionaries
    """
    config = loaded_model.config

    # Trainer config - extract from loaded model, matching template defaults
    trainer_defaults = {
        "auto_lr_find": False,
        "batch_size": min(128, max(32, n_train // 16)),
        "max_epochs": 100,
        "min_epochs": 10,
        "early_stopping": "valid_loss",
        "early_stopping_patience": 10,
        "gradient_clip_val": 1.0,
    }

    trainer_config = {}
    for key, default in trainer_defaults.items():
        value = getattr(config, key, default)
        if value == default and not hasattr(config, key):
            log.warning(f"Trainer config '{key}' not found in loaded model, using default: {default}")
        trainer_config[key] = value

    # Model config - extract from loaded model, matching template defaults
    model_defaults = {
        "layers": "256-128-64",
        "activation": "LeakyReLU",
        "learning_rate": 1e-3,
        "dropout": 0.3,
        "use_batch_norm": True,
        "initialization": "kaiming",
    }

    model_config = {}
    for key, default in model_defaults.items():
        value = getattr(config, key, default)
        if value == default and not hasattr(config, key):
            log.warning(f"Model config '{key}' not found in loaded model, using default: {default}")
        model_config[key] = value

    return {"trainer": trainer_config, "model": model_config}


def pull_cv_results(workbench_model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull cross-validation results from AWS training artifacts.

    This retrieves the validation predictions and training metrics that were
    saved during model training.

    Args:
        workbench_model: Workbench model object

    Returns:
        Tuple of:
            - DataFrame with training metrics
            - DataFrame with validation predictions
    """
    # Get the validation predictions from S3
    s3_path = f"{workbench_model.model_training_path}/validation_predictions.csv"
    predictions_df = pull_s3_data(s3_path)

    if predictions_df is None:
        raise ValueError(f"No validation predictions found at {s3_path}")

    log.info(f"Pulled {len(predictions_df)} validation predictions from {s3_path}")

    # Get training metrics from model metadata
    training_metrics = workbench_model.workbench_meta().get("workbench_training_metrics")

    if training_metrics is None:
        log.warning(f"No training metrics found in model metadata for {workbench_model.model_name}")
        metrics_df = pd.DataFrame({"error": [f"No training metrics found for {workbench_model.model_name}"]})
    else:
        metrics_df = pd.DataFrame.from_dict(training_metrics)
        log.info(f"Metrics summary:\n{metrics_df.to_string(index=False)}")

    return metrics_df, predictions_df


def cross_fold_inference(
    workbench_model: Any,
    nfolds: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Performs K-fold cross-validation for PyTorch Tabular models.

    Replicates the training setup from the original model to ensure
    cross-validation results are comparable to the deployed model.

    Args:
        workbench_model: Workbench model object
        nfolds: Number of folds for cross-validation (default is 5)

    Returns:
        Tuple of:
            - DataFrame with per-class metrics (and 'all' row for overall metrics)
            - DataFrame with columns: id, target, prediction, and *_proba columns (for classifiers)
    """
    import shutil

    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import CategoryEmbeddingModelConfig

    from workbench.api import FeatureSet

    # Create a temporary model directory
    model_dir = tempfile.mkdtemp(prefix="pytorch_cv_")
    log.info(f"Using model directory: {model_dir}")

    try:
        # Download and extract model artifacts to get config and artifacts
        model_artifact_uri = workbench_model.model_data_url()
        download_and_extract_model(model_artifact_uri, model_dir)

        # Load model and artifacts
        loaded_model, artifacts = load_pytorch_model_artifacts(model_dir)
        category_mappings = artifacts.get("category_mappings", {})

        # Determine if classifier from the loaded model's config
        is_classifier = loaded_model.config.task == "classification"

        # Use saved label encoder if available, otherwise create fresh one
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
                    df[col] = pd.Categorical(df[col], categories=category_mappings[col])
                else:
                    df[col] = df[col].astype("category")

        # Determine categorical and continuous columns
        categorical_cols = [col for col in feature_cols if df[col].dtype.name == "category"]
        continuous_cols = [col for col in feature_cols if col not in categorical_cols]

        # Cast continuous columns to float
        if continuous_cols:
            df[continuous_cols] = df[continuous_cols].astype("float64")

        # Drop rows with NaN features or target (PyTorch Tabular cannot handle NaN values)
        nan_mask = df[feature_cols].isna().any(axis=1) | df[target_col].isna()
        if nan_mask.any():
            n_nan_rows = nan_mask.sum()
            log.warning(
                f"Dropping {n_nan_rows} rows ({100*n_nan_rows/len(df):.1f}%) with NaN values for cross-validation"
            )
            df = df[~nan_mask].reset_index(drop=True)

        X = df[feature_cols]
        y = df[target_col]
        ids = df[id_col]

        # Encode target if classifier
        if label_encoder is not None:
            if not hasattr(label_encoder, "classes_"):
                label_encoder.fit(y)
            y_encoded = label_encoder.transform(y)
            y_for_cv = pd.Series(y_encoded, index=y.index, name=target_col)
        else:
            y_for_cv = y

        # Extract configs from loaded model (pass approx train size for batch_size calculation)
        n_train_approx = int(len(df) * (1 - 1 / nfolds))
        configs = _extract_model_configs(loaded_model, n_train_approx)
        trainer_params = configs["trainer"]
        model_params = configs["model"]

        log.info(f"Trainer config:\n{pformat(trainer_params)}")
        log.info(f"Model config:\n{pformat(model_params)}")

        # Prepare KFold
        kfold = (StratifiedKFold if is_classifier else KFold)(n_splits=nfolds, shuffle=True, random_state=42)

        # Initialize results collection
        fold_metrics = []
        predictions_df = pd.DataFrame({id_col: ids, target_col: y})
        if is_classifier:
            predictions_df["pred_proba"] = [None] * len(predictions_df)

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

            # Create configs for this fold - matching the training template exactly
            data_config = DataConfig(
                target=[target_col],
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols,
            )

            trainer_config = TrainerConfig(
                auto_lr_find=trainer_params["auto_lr_find"],
                batch_size=trainer_params["batch_size"],
                max_epochs=trainer_params["max_epochs"],
                min_epochs=trainer_params["min_epochs"],
                early_stopping=trainer_params["early_stopping"],
                early_stopping_patience=trainer_params["early_stopping_patience"],
                gradient_clip_val=trainer_params["gradient_clip_val"],
                checkpoints="valid_loss",  # Save best model based on validation loss
                accelerator="cpu",
            )

            optimizer_config = OptimizerConfig()

            model_config = CategoryEmbeddingModelConfig(
                task="classification" if is_classifier else "regression",
                layers=model_params["layers"],
                activation=model_params["activation"],
                learning_rate=model_params["learning_rate"],
                dropout=model_params["dropout"],
                use_batch_norm=model_params["use_batch_norm"],
                initialization=model_params["initialization"],
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
                # Clean up checkpoint directory from previous fold
                checkpoint_dir = "/tmp/saved_models"
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                tabular_model.fit(train=df_train, validation=df_val)
            finally:
                os.chdir(original_cwd)

            # Make predictions
            result = tabular_model.predict(df_val[feature_cols])

            # Extract predictions
            prediction_col = f"{target_col}_prediction"
            preds = result[prediction_col].values

            # Store predictions at the correct indices
            val_indices = df.iloc[val_idx].index
            if is_classifier:
                preds_decoded = label_encoder.inverse_transform(preds.astype(int))
                predictions_df.loc[val_indices, "prediction"] = preds_decoded

                # Get probabilities and store at validation indices only
                prob_cols = sorted([col for col in result.columns if col.endswith("_probability")])
                if prob_cols:
                    probs = result[prob_cols].values
                    for i, idx in enumerate(val_indices):
                        predictions_df.at[idx, "pred_proba"] = probs[i].tolist()
            else:
                predictions_df.loc[val_indices, "prediction"] = preds

            # Calculate fold metrics
            if is_classifier:
                y_val_orig = label_encoder.inverse_transform(df_val[target_col])
                preds_orig = preds_decoded

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average="weighted", zero_division=0
                )

                prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average=None, zero_division=0, labels=label_encoder.classes_
                )

                y_val_encoded = df_val[target_col].values
                roc_auc_overall = roc_auc_score(y_val_encoded, probs, multi_class="ovr", average="macro")
                roc_auc_per_class = roc_auc_score(y_val_encoded, probs, multi_class="ovr", average=None)

                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": roc_auc_overall,
                        "precision_per_class": prec_per_class,
                        "recall_per_class": rec_per_class,
                        "f1_per_class": f1_per_class,
                        "roc_auc_per_class": roc_auc_per_class,
                    }
                )

                print(f"Fold {fold_idx} - F1: {f1:.4f}, ROC-AUC: {roc_auc_overall:.4f}")
            else:
                y_val = df_val[target_col].values
                spearman_corr, _ = spearmanr(y_val, preds)
                rmse = np.sqrt(mean_squared_error(y_val, preds))

                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "rmse": rmse,
                        "mae": mean_absolute_error(y_val, preds),
                        "medae": median_absolute_error(y_val, preds),
                        "r2": r2_score(y_val, preds),
                        "spearmanr": spearman_corr,
                    }
                )

                print(f"Fold {fold_idx} - RMSE: {rmse:.4f}, R2: {fold_metrics[-1]['r2']:.4f}")

        # Calculate summary metrics
        fold_df = pd.DataFrame(fold_metrics)

        if is_classifier:
            if "pred_proba" in predictions_df.columns:
                predictions_df = expand_proba_column(predictions_df, label_encoder.classes_)

            metric_rows = []
            for idx, class_name in enumerate(label_encoder.classes_):
                prec_scores = np.array([fold["precision_per_class"][idx] for fold in fold_metrics])
                rec_scores = np.array([fold["recall_per_class"][idx] for fold in fold_metrics])
                f1_scores = np.array([fold["f1_per_class"][idx] for fold in fold_metrics])
                roc_auc_scores = np.array([fold["roc_auc_per_class"][idx] for fold in fold_metrics])

                y_orig = label_encoder.inverse_transform(y_for_cv)
                support = int((y_orig == class_name).sum())

                metric_rows.append(
                    {
                        "class": class_name,
                        "precision": prec_scores.mean(),
                        "recall": rec_scores.mean(),
                        "f1": f1_scores.mean(),
                        "roc_auc": roc_auc_scores.mean(),
                        "support": support,
                    }
                )

            metric_rows.append(
                {
                    "class": "all",
                    "precision": fold_df["precision"].mean(),
                    "recall": fold_df["recall"].mean(),
                    "f1": fold_df["f1"].mean(),
                    "roc_auc": fold_df["roc_auc"].mean(),
                    "support": len(y_for_cv),
                }
            )

            metrics_df = pd.DataFrame(metric_rows)
        else:
            metrics_df = pd.DataFrame(
                [
                    {
                        "rmse": fold_df["rmse"].mean(),
                        "mae": fold_df["mae"].mean(),
                        "medae": fold_df["medae"].mean(),
                        "r2": fold_df["r2"].mean(),
                        "spearmanr": fold_df["spearmanr"].mean(),
                        "support": len(y_for_cv),
                    }
                ]
            )

        print(f"\n{'='*50}")
        print("Cross-Validation Summary")
        print(f"{'='*50}")
        print(metrics_df.to_string(index=False))

        return metrics_df, predictions_df

    finally:
        log.info(f"Cleaning up model directory: {model_dir}")
        shutil.rmtree(model_dir, ignore_errors=True)


if __name__ == "__main__":

    # Tests for the PyTorch utilities
    from workbench.api import Model, Endpoint

    # Initialize Workbench model
    model_name = "caco2-er-reg-pytorch-test"
    # model_name = "aqsol-pytorch-reg"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Perform cross-fold inference
    end = Endpoint(model.endpoints()[0])
    end.cross_fold_inference()
