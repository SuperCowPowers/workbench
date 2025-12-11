"""ChemProp utilities for Workbench models."""

# flake8: noqa: E402
import logging
import os
import tempfile
from typing import Any, Tuple

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


def load_chemprop_model_artifacts(model_dir: str) -> Tuple[Any, dict]:
    """Load ChemProp MPNN model and artifacts from an extracted model directory.

    Args:
        model_dir: Directory containing extracted model artifacts

    Returns:
        Tuple of (MPNN model, artifacts_dict).
        artifacts_dict contains 'label_encoder' and 'feature_metadata' if present.
    """
    import joblib
    from chemprop import models

    model_path = os.path.join(model_dir, "chemprop_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No chemprop_model.pt found in {model_dir}")

    model = models.MPNN.load_from_file(model_path)
    model.eval()

    # Load additional artifacts
    artifacts = {}

    label_encoder_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.exists(label_encoder_path):
        artifacts["label_encoder"] = joblib.load(label_encoder_path)

    feature_metadata_path = os.path.join(model_dir, "feature_metadata.joblib")
    if os.path.exists(feature_metadata_path):
        artifacts["feature_metadata"] = joblib.load(feature_metadata_path)

    return model, artifacts


def _find_smiles_column(columns: list) -> str:
    """Find the SMILES column name from a list (case-insensitive match for 'smiles')."""
    smiles_column = next((col for col in columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Column list must contain a 'smiles' column (case-insensitive)")
    return smiles_column


def _create_molecule_datapoints(
    smiles_list: list,
    targets: list = None,
    extra_descriptors: np.ndarray = None,
) -> Tuple[list, list]:
    """Create ChemProp MoleculeDatapoints from SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        targets: Optional list of target values (for training)
        extra_descriptors: Optional array of extra features (n_samples, n_features)

    Returns:
        Tuple of (list of MoleculeDatapoint objects, list of valid indices)
    """
    from chemprop import data
    from rdkit import Chem

    datapoints = []
    valid_indices = []
    invalid_count = 0

    for i, smi in enumerate(smiles_list):
        # Validate SMILES with RDKit first
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_count += 1
            continue

        # Build datapoint with optional target and extra descriptors
        y = [targets[i]] if targets is not None else None
        x_d = extra_descriptors[i] if extra_descriptors is not None else None

        dp = data.MoleculeDatapoint.from_smi(smi, y=y, x_d=x_d)
        datapoints.append(dp)
        valid_indices.append(i)

    if invalid_count > 0:
        print(f"Warning: Skipped {invalid_count} invalid SMILES strings")

    return datapoints, valid_indices


def _build_mpnn_model(
    hyperparameters: dict,
    task: str = "regression",
    num_classes: int = None,
    n_extra_descriptors: int = 0,
    x_d_transform: Any = None,
    output_transform: Any = None,
) -> Any:
    """Build an MPNN model with the specified hyperparameters.

    Args:
        hyperparameters: Dictionary of model hyperparameters
        task: Either "regression" or "classification"
        num_classes: Number of classes for classification tasks
        n_extra_descriptors: Number of extra descriptor features (for hybrid mode)
        x_d_transform: Optional transform for extra descriptors (scaling)
        output_transform: Optional transform for regression output (unscaling targets)

    Returns:
        Configured MPNN model
    """
    from chemprop import models, nn

    # Model hyperparameters with defaults
    hidden_dim = hyperparameters.get("hidden_dim", 300)
    depth = hyperparameters.get("depth", 3)
    dropout = hyperparameters.get("dropout", 0.1)
    ffn_hidden_dim = hyperparameters.get("ffn_hidden_dim", 300)
    ffn_num_layers = hyperparameters.get("ffn_num_layers", 1)

    # Message passing component
    mp = nn.BondMessagePassing(d_h=hidden_dim, depth=depth, dropout=dropout)

    # Aggregation - NormAggregation normalizes output, recommended when using extra descriptors
    agg = nn.NormAggregation()

    # FFN input_dim = message passing output + extra descriptors
    ffn_input_dim = hidden_dim + n_extra_descriptors

    # Build FFN based on task type
    if task == "classification" and num_classes is not None:
        # Multi-class classification
        ffn = nn.MulticlassClassificationFFN(
            n_classes=num_classes,
            input_dim=ffn_input_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_num_layers,
            dropout=dropout,
        )
    else:
        # Regression with optional output transform to unscale predictions
        ffn = nn.RegressionFFN(
            input_dim=ffn_input_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=ffn_num_layers,
            dropout=dropout,
            output_transform=output_transform,
        )

    # Create the MPNN model
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=True,
        metrics=None,
        X_d_transform=x_d_transform,
    )

    return mpnn


def _extract_model_hyperparameters(loaded_model: Any) -> dict:
    """Extract hyperparameters from a loaded ChemProp MPNN model.

    Extracts architecture parameters from the model's components to replicate
    the exact same model configuration during cross-validation.

    Args:
        loaded_model: Loaded MPNN model instance

    Returns:
        Dictionary of hyperparameters matching the training template
    """
    hyperparameters = {}

    # Extract from message passing layer (BondMessagePassing)
    mp = loaded_model.message_passing
    hyperparameters["hidden_dim"] = getattr(mp, "d_h", 300)
    hyperparameters["depth"] = getattr(mp, "depth", 3)

    # Dropout is stored as a nn.Dropout module, get the p value
    if hasattr(mp, "dropout"):
        dropout_module = mp.dropout
        hyperparameters["dropout"] = getattr(dropout_module, "p", 0.0)
    else:
        hyperparameters["dropout"] = 0.0

    # Extract from predictor (FFN - either RegressionFFN or MulticlassClassificationFFN)
    ffn = loaded_model.predictor

    # FFN hidden_dim - try multiple attribute names
    if hasattr(ffn, "hidden_dim"):
        hyperparameters["ffn_hidden_dim"] = ffn.hidden_dim
    elif hasattr(ffn, "d_h"):
        hyperparameters["ffn_hidden_dim"] = ffn.d_h
    else:
        hyperparameters["ffn_hidden_dim"] = 300

    # FFN num_layers - try multiple attribute names
    if hasattr(ffn, "n_layers"):
        hyperparameters["ffn_num_layers"] = ffn.n_layers
    elif hasattr(ffn, "num_layers"):
        hyperparameters["ffn_num_layers"] = ffn.num_layers
    else:
        hyperparameters["ffn_num_layers"] = 1

    # Training hyperparameters (use defaults matching the template)
    hyperparameters["max_epochs"] = 50
    hyperparameters["patience"] = 10

    return hyperparameters


def _get_n_extra_descriptors(loaded_model: Any) -> int:
    """Get the number of extra descriptors from the loaded model.

    The model's X_d_transform contains the scaler which knows the feature dimension.

    Args:
        loaded_model: Loaded MPNN model instance

    Returns:
        Number of extra descriptors (0 if none)
    """
    x_d_transform = loaded_model.X_d_transform
    if x_d_transform is None:
        return 0

    # ScaleTransform wraps a StandardScaler, check its mean_ attribute
    if hasattr(x_d_transform, "mean"):
        # x_d_transform.mean is a tensor
        return len(x_d_transform.mean)
    elif hasattr(x_d_transform, "scaler") and hasattr(x_d_transform.scaler, "mean_"):
        return len(x_d_transform.scaler.mean_)

    return 0


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
    """Performs K-fold cross-validation for ChemProp MPNN models.

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

    import joblib
    import torch
    from chemprop import data, nn
    from lightning import pytorch as pl

    from workbench.api import FeatureSet

    # Create a temporary model directory
    model_dir = tempfile.mkdtemp(prefix="chemprop_cv_")
    log.info(f"Using model directory: {model_dir}")

    try:
        # Download and extract model artifacts to get config and artifacts
        model_artifact_uri = workbench_model.model_data_url()
        download_and_extract_model(model_artifact_uri, model_dir)

        # Load model and artifacts
        loaded_model, artifacts = load_chemprop_model_artifacts(model_dir)
        feature_metadata = artifacts.get("feature_metadata", {})

        # Determine if classifier from predictor type
        from chemprop.nn import MulticlassClassificationFFN

        is_classifier = isinstance(loaded_model.predictor, MulticlassClassificationFFN)

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

        # Find SMILES column
        smiles_column = _find_smiles_column(feature_cols)

        # Determine extra feature columns:
        # 1. First try feature_metadata (saved during training)
        # 2. Fall back to inferring from feature_cols (exclude SMILES column)
        # 3. Verify against model's X_d_transform dimension
        if feature_metadata and "extra_feature_cols" in feature_metadata:
            extra_feature_cols = feature_metadata["extra_feature_cols"]
        else:
            # Infer from feature list - everything except SMILES is an extra feature
            extra_feature_cols = [f for f in feature_cols if f.lower() != "smiles"]

            # Verify against model's actual extra descriptor dimension
            n_extra_from_model = _get_n_extra_descriptors(loaded_model)
            if n_extra_from_model > 0 and len(extra_feature_cols) != n_extra_from_model:
                log.warning(
                    f"Inferred {len(extra_feature_cols)} extra features but model expects "
                    f"{n_extra_from_model}. Using inferred columns."
                )

        use_extra_features = len(extra_feature_cols) > 0

        print(f"SMILES column: {smiles_column}")
        print(f"Extra features: {extra_feature_cols if use_extra_features else 'None (SMILES only)'}")

        # Drop rows with missing SMILES or target values
        initial_count = len(df)
        df = df.dropna(subset=[smiles_column, target_col])
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"Dropped {dropped} rows with missing SMILES or target values")

        # Extract hyperparameters from loaded model
        hyperparameters = _extract_model_hyperparameters(loaded_model)
        print(f"Extracted hyperparameters: {hyperparameters}")

        # Get number of classes for classifier
        num_classes = None
        if is_classifier:
            # Try to get from loaded model's FFN first (most reliable)
            ffn = loaded_model.predictor
            if hasattr(ffn, "n_classes"):
                num_classes = ffn.n_classes
            elif label_encoder is not None and hasattr(label_encoder, "classes_"):
                num_classes = len(label_encoder.classes_)
            else:
                # Fit label encoder to get classes
                if label_encoder is None:
                    label_encoder = LabelEncoder()
                label_encoder.fit(df[target_col])
                num_classes = len(label_encoder.classes_)
            print(f"Classification task with {num_classes} classes")

        X = df[[smiles_column] + extra_feature_cols]
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

            # Prepare extra features if using hybrid mode
            train_extra_features = None
            val_extra_features = None
            col_means = None

            if use_extra_features:
                train_extra_features = df_train[extra_feature_cols].values.astype(np.float32)
                val_extra_features = df_val[extra_feature_cols].values.astype(np.float32)

                # Fill NaN with column means from training data
                col_means = np.nanmean(train_extra_features, axis=0)
                for i in range(train_extra_features.shape[1]):
                    train_nan_mask = np.isnan(train_extra_features[:, i])
                    val_nan_mask = np.isnan(val_extra_features[:, i])
                    train_extra_features[train_nan_mask, i] = col_means[i]
                    val_extra_features[val_nan_mask, i] = col_means[i]

            # Create ChemProp datasets
            train_datapoints, train_valid_idx = _create_molecule_datapoints(
                df_train[smiles_column].tolist(),
                df_train[target_col].tolist(),
                train_extra_features,
            )
            val_datapoints, val_valid_idx = _create_molecule_datapoints(
                df_val[smiles_column].tolist(),
                df_val[target_col].tolist(),
                val_extra_features,
            )

            # Update dataframes to only include valid molecules
            df_train_valid = df_train.iloc[train_valid_idx].reset_index(drop=True)
            df_val_valid = df_val.iloc[val_valid_idx].reset_index(drop=True)

            train_dataset = data.MoleculeDataset(train_datapoints)
            val_dataset = data.MoleculeDataset(val_datapoints)

            # Save raw validation features before scaling
            val_extra_raw = val_extra_features[val_valid_idx] if val_extra_features is not None else None

            # Scale extra descriptors
            feature_scaler = None
            x_d_transform = None
            if use_extra_features:
                feature_scaler = train_dataset.normalize_inputs("X_d")
                val_dataset.normalize_inputs("X_d", feature_scaler)
                x_d_transform = nn.ScaleTransform.from_standard_scaler(feature_scaler)

            # Scale targets for regression
            target_scaler = None
            output_transform = None
            if not is_classifier:
                target_scaler = train_dataset.normalize_targets()
                val_dataset.normalize_targets(target_scaler)
                output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)

            # Get batch size
            batch_size = min(64, max(16, len(df_train_valid) // 16))

            train_loader = data.build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = data.build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

            # Build the model
            n_extra = len(extra_feature_cols) if use_extra_features else 0
            mpnn = _build_mpnn_model(
                hyperparameters,
                task="classification" if is_classifier else "regression",
                num_classes=num_classes,
                n_extra_descriptors=n_extra,
                x_d_transform=x_d_transform,
                output_transform=output_transform,
            )

            # Training configuration
            max_epochs = hyperparameters.get("max_epochs", 50)
            patience = hyperparameters.get("patience", 10)

            # Set up trainer
            checkpoint_dir = os.path.join(model_dir, f"fold_{fold_idx}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            callbacks = [
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
                pl.callbacks.ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename="best_model",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                ),
            ]

            trainer = pl.Trainer(
                accelerator="auto",
                max_epochs=max_epochs,
                callbacks=callbacks,
                logger=False,
                enable_progress_bar=True,
            )

            # Train the model
            trainer.fit(mpnn, train_loader, val_loader)

            # Load the best checkpoint
            if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
                best_ckpt_path = trainer.checkpoint_callback.best_model_path
                checkpoint = torch.load(best_ckpt_path, weights_only=False)
                mpnn.load_state_dict(checkpoint["state_dict"])

            mpnn.eval()

            # Make predictions using raw features
            val_datapoints_raw, _ = _create_molecule_datapoints(
                df_val_valid[smiles_column].tolist(),
                df_val_valid[target_col].tolist(),
                val_extra_raw,
            )
            val_dataset_raw = data.MoleculeDataset(val_datapoints_raw)
            val_loader_pred = data.build_dataloader(val_dataset_raw, batch_size=batch_size, shuffle=False)

            with torch.inference_mode():
                val_predictions = trainer.predict(mpnn, val_loader_pred)

            preds = np.concatenate([p.numpy() for p in val_predictions], axis=0)

            # ChemProp may return (n_samples, 1, n_classes) for multiclass - squeeze middle dim
            if preds.ndim == 3 and preds.shape[1] == 1:
                preds = preds.squeeze(axis=1)

            # Map predictions back to original indices
            original_val_indices = df.iloc[val_idx].index[val_valid_idx]

            if is_classifier:
                # Get class predictions
                if preds.ndim == 2 and preds.shape[1] > 1:
                    class_preds = np.argmax(preds, axis=1)
                else:
                    class_preds = (preds.flatten() > 0.5).astype(int)

                preds_decoded = label_encoder.inverse_transform(class_preds)
                predictions_df.loc[original_val_indices, "prediction"] = preds_decoded

                # Store probabilities
                if preds.ndim == 2 and preds.shape[1] > 1:
                    for i, idx in enumerate(original_val_indices):
                        predictions_df.at[idx, "pred_proba"] = preds[i].tolist()
            else:
                predictions_df.loc[original_val_indices, "prediction"] = preds.flatten()

            # Calculate fold metrics
            y_val = df_val_valid[target_col].values

            if is_classifier:
                y_val_orig = label_encoder.inverse_transform(y_val.astype(int))
                preds_orig = preds_decoded

                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average="weighted", zero_division=0
                )

                prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
                    y_val_orig, preds_orig, average=None, zero_division=0, labels=label_encoder.classes_
                )

                # ROC AUC
                if preds.ndim == 2 and preds.shape[1] > 1:
                    roc_auc_overall = roc_auc_score(y_val, preds, multi_class="ovr", average="macro")
                    roc_auc_per_class = roc_auc_score(y_val, preds, multi_class="ovr", average=None)
                else:
                    roc_auc_overall = roc_auc_score(y_val, preds.flatten())
                    roc_auc_per_class = [roc_auc_overall]

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
                spearman_corr, _ = spearmanr(y_val, preds.flatten())
                rmse = np.sqrt(mean_squared_error(y_val, preds.flatten()))

                fold_metrics.append(
                    {
                        "fold": fold_idx,
                        "rmse": rmse,
                        "mae": mean_absolute_error(y_val, preds.flatten()),
                        "medae": median_absolute_error(y_val, preds.flatten()),
                        "r2": r2_score(y_val, preds.flatten()),
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

    # Tests for the ChemProp utilities
    from workbench.api import Endpoint, Model

    # Initialize Workbench model
    model_name = "aqsol-chemprop-reg"
    print(f"Loading Workbench model: {model_name}")
    model = Model(model_name)
    print(f"Model Framework: {model.model_framework}")

    # Perform cross-fold inference
    end = Endpoint(model.endpoints()[0])
    end.cross_fold_inference()
