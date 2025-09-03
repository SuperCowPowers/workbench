# Imports for Chemprop Model
import os
import awswrangler as wr
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from chemprop import data, featurizers, models, nn
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

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
from io import StringIO
import json
import argparse
import joblib
import os
import pandas as pd
from typing import List, Tuple


# Set RDKit logger to only show errors or critical messages
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

# Template Parameters
TEMPLATE_PARAMS = {
    # "model_type": "regressor",
    "model_type": "classifier",
    # "target_column": "solubility",
    "target_column": "solubility_class",
    "smiles_column": "smiles",  # New parameter for SMILES column
    "model_metrics_s3_path": "",
    "train_all_data": False,
    "hyperparameters": {},
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
    return df


def create_chemprop_model(model_type: str, hyperparameters: dict, n_classes: int = 1):
    """
    Create a Chemprop MPNN model based on the specified type and hyperparameters.

    Args:
        model_type (str): "classifier" or "regressor"
        hyperparameters (dict): Hyperparameters for the model
        n_classes (int): Number of classification classes (default 1)

    Returns:
        models.MPNN: Configured Chemprop model
    """

    # Message passing
    mp = nn.BondMessagePassing(
        depth=hyperparameters.get("depth", 3),
        dropout=hyperparameters.get("dropout", 0.0),
    )

    # Aggregation
    agg_type = hyperparameters.get("aggregation", "mean")
    if agg_type == "mean":
        agg = nn.MeanAggregation()
    elif agg_type == "sum":
        agg = nn.SumAggregation()
    else:
        agg = nn.MeanAggregation()  # Default fallback

    # Feed-forward network based on model type
    if model_type == "classifier":
        if n_classes > 2:
            ffn = nn.MulticlassClassificationFFN(n_classes=n_classes)
            metric_list = [nn.metrics.BinaryAccuracy()]  # Will need to adjust for multiclass
        else:
            ffn = nn.BinaryClassificationFFN()
            metric_list = [
                nn.metrics.BinaryAUROC(),
                nn.metrics.BinaryAUPRC(),
                nn.metrics.BinaryAccuracy(),
                nn.metrics.BinaryF1Score(),
            ]
    else:
        ffn = nn.RegressionFFN()
        metric_list = [nn.metrics.RMSE(), nn.metrics.MAE(), nn.metrics.R2Score()]

    # Create MPNN
    mpnn = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=hyperparameters.get("batch_norm", True),
        metrics=metric_list,
    )

    return mpnn


def model_fn(model_dir):
    """Deserialize and return fitted Chemprop model"""
    model_path = os.path.join(model_dir, "chemprop_model.ckpt")

    # Load model parameters to recreate the model
    with open(os.path.join(model_dir, "model_config.json")) as fp:
        config = json.load(fp)

    model = create_chemprop_model(config["model_type"], config["hyperparameters"], config.get("n_classes", 1))

    # Load the trained weights
    model = models.MPNN.load_from_checkpoint(model_path)
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
    """Make Predictions with our Chemprop Model
    Args:
        df (pd.DataFrame): The input DataFrame with SMILES strings
        model: The Chemprop model used for predictions
    Returns:
        pd.DataFrame: The DataFrame with the predictions added
    """

    # Load model configuration
    model_dir = os.environ.get("SM_MODEL_DIR", "chemprop_outputs")
    with open(os.path.join(model_dir, "model_config.json")) as fp:
        config = json.load(fp)

    smiles_column = config["smiles_column"]
    target_column = config["target_column"]
    model_type = config["model_type"]

    # Load label encoder if available
    label_encoder = None
    if os.path.exists(os.path.join(model_dir, "label_encoder.joblib")):
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))

    # Validate SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in input data")

    # Create molecule datapoints
    smiles_list = df[smiles_column].tolist()

    # For prediction, we don't have targets, so use None
    molecule_data = [MoleculeDatapoint.from_smi(smi, y=None) for smi in smiles_list]

    # Create dataset and dataloader
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    test_dataset = MoleculeDataset(molecule_data, featurizer)
    test_loader = build_dataloader(test_dataset, batch_size=64, shuffle=False)

    # Make predictions
    trainer = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="cpu", devices=1)

    with torch.inference_mode():
        predictions = trainer.predict(model, test_loader)

    # Process predictions
    if predictions:
        # Concatenate all batch predictions
        all_preds = torch.cat(predictions, dim=0)

        if model_type == "classifier":
            if label_encoder and all_preds.dim() > 1:  # Multi-class or binary with probabilities
                # Get class predictions and probabilities
                if all_preds.shape[1] == 1:  # Binary classification
                    probs = torch.sigmoid(all_preds).squeeze().numpy()
                    preds = (probs > 0.5).astype(int)
                    df["pred_proba"] = [[1 - p, p] for p in probs]  # [negative_prob, positive_prob]
                else:  # Multi-class
                    probs = torch.softmax(all_preds, dim=1).numpy()
                    preds = np.argmax(probs, axis=1)
                    df["pred_proba"] = probs.tolist()

                # Decode predictions using label encoder
                df["prediction"] = label_encoder.inverse_transform(preds)

                # Expand probabilities into separate columns
                df = expand_proba_column(df, label_encoder.classes_)
            else:
                # Simple binary case
                probs = torch.sigmoid(all_preds).squeeze().numpy()
                df["prediction"] = (probs > 0.5).astype(int)
        else:
            # Regression
            df["prediction"] = all_preds.squeeze().numpy()

    return df


if __name__ == "__main__":
    """The main function is for training the Chemprop model"""
    # Harness Template Parameters
    target = TEMPLATE_PARAMS["target_column"]
    smiles_column = TEMPLATE_PARAMS["smiles_column"]
    model_type = TEMPLATE_PARAMS["model_type"]
    model_metrics_s3_path = TEMPLATE_PARAMS["model_metrics_s3_path"]
    train_all_data = TEMPLATE_PARAMS["train_all_data"]
    hyperparameters = TEMPLATE_PARAMS["hyperparameters"]
    validation_split = 0.2
    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "chemprop_outputs"))
    args = parser.parse_args()
    # Pull training data from a FeatureSet
    from workbench.api import FeatureSet

    fs = FeatureSet("aqsol_features")
    all_df = fs.pull_dataframe()
    # Check if the dataframe is empty
    check_dataframe(all_df, "training_df")
    # Validate required columns
    if smiles_column not in all_df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in data")
    if target not in all_df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    print(f"Target: {target}")
    print(f"SMILES Column: {smiles_column}")

    # Ensure that all smiles will convert to valid molecules
    valid_mask = all_df[smiles_column].apply(lambda x: Chem.MolFromSmiles(x) is not None)
    # Log failed conversions
    failed_smiles = all_df[~valid_mask][smiles_column].tolist()
    if failed_smiles:
        print(f"Failed to convert the following SMILES to molecules: {failed_smiles}")
    # Keep only rows with valid SMILES
    all_df = all_df[valid_mask]

    # Split data
    if train_all_data:
        print("Training on ALL of the data")
        df_train = all_df.copy()
        df_val = all_df.copy()
    elif "training" in all_df.columns:
        print("Found training column, splitting data based on training column")
        df_train = all_df[all_df["training"]]
        df_val = all_df[~all_df["training"]]
    else:
        from sklearn.model_selection import train_test_split

        print("WARNING: No training column found, splitting data with random state=42")
        df_train, df_val = train_test_split(all_df, test_size=validation_split, random_state=42)

    print(f"FIT/TRAIN: {df_train.shape}")
    print(f"VALIDATION: {df_val.shape}")
    # Set up label encoder for classification
    label_encoder = None
    n_classes = 1
    if model_type == "classifier":
        label_encoder = LabelEncoder()
        df_train[target] = label_encoder.fit_transform(df_train[target])
        df_val[target] = label_encoder.transform(df_val[target])
        n_classes = len(label_encoder.classes_)
        print(f"Classification with {n_classes} classes: {label_encoder.classes_}")

    # Create molecule datapoints
    print("Creating molecule datapoints...")
    train_data = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df_train[smiles_column], df_train[target])]
    val_data = [MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(df_val[smiles_column], df_val[target])]
    # Create datasets
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    train_dataset = MoleculeDataset(train_data, featurizer)
    scaler = train_dataset.normalize_targets() if model_type == "regressor" else None

    val_dataset = MoleculeDataset(val_data, featurizer)
    if scaler:
        val_dataset.normalize_targets(scaler)
    # Create dataloaders
    batch_size = hyperparameters.get("batch_size", 64)
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, num_workers=0)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    print("Creating Chemprop model...")
    model = create_chemprop_model(model_type, hyperparameters, n_classes)
    # Set up trainer with callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, save_last=True)
    trainer = pl.Trainer(
        max_epochs=hyperparameters.get("max_epochs", 100),
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
    )
    # Train the model
    print("Training Chemprop model...")
    trainer.fit(model, train_loader, val_loader)
    # Load best model for evaluation
    best_model_path = checkpoint_callback.best_model_path
    best_model = models.MPNN.load_from_checkpoint(best_model_path)
    # Make predictions on validation set
    print("Making predictions on validation set...")
    trainer_eval = pl.Trainer(logger=False, enable_progress_bar=False, accelerator="auto", devices=1)

    with torch.inference_mode():
        predictions = trainer_eval.predict(best_model, val_loader)
    # Process predictions for evaluation
    if predictions:
        all_preds = torch.cat(predictions, dim=0)

        if model_type == "classifier":
            if all_preds.shape[1] == 1:  # Binary classification
                probs = torch.sigmoid(all_preds).squeeze().numpy()
                preds = (probs > 0.5).astype(int)
                df_val["pred_proba"] = [[1 - p, p] for p in probs]
            else:  # Multi-class
                probs = torch.softmax(all_preds, dim=1).numpy()
                preds = np.argmax(probs, axis=1)
                df_val["pred_proba"] = probs.tolist()

            # Set predictions and handle label encoding
            if label_encoder:
                y_validate = label_encoder.inverse_transform(df_val[target])
                preds_decoded = label_encoder.inverse_transform(preds)
                df_val["prediction"] = preds_decoded

                # Expand probabilities
                df_val = expand_proba_column(df_val, label_encoder.classes_)
            else:
                y_validate = df_val[target].values
                df_val["prediction"] = preds
        else:
            # Regression
            preds = all_preds.squeeze().numpy()

            # Denormalize if we used scaling
            if scaler:
                preds = scaler.denormalize(torch.tensor(preds)).numpy()

            df_val["prediction"] = preds
            y_validate = df_val[target].values
    # Save predictions to S3
    output_columns = [target, "prediction"]
    if model_type == "classifier":
        output_columns += [col for col in df_val.columns if col.endswith("_proba")]

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
        scores = precision_recall_fscore_support(y_validate, preds_decoded, average=None, labels=label_names)
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
        conf_mtx = confusion_matrix(y_validate, preds_decoded, labels=label_names)
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
    # Save the model and configuration
    os.makedirs(args.model_dir, exist_ok=True)

    # Save the best model checkpoint
    import shutil

    shutil.copy(best_model_path, os.path.join(args.model_dir, "chemprop_model.ckpt"))
    # Save model configuration
    model_config = {
        "model_type": model_type,
        "target_column": target,
        "smiles_column": smiles_column,
        "hyperparameters": hyperparameters,
        "n_classes": n_classes,
    }

    with open(os.path.join(args.model_dir, "model_config.json"), "w") as fp:
        json.dump(model_config, fp)
    # Save label encoder if we have one
    if label_encoder:
        joblib.dump(label_encoder, os.path.join(args.model_dir, "label_encoder.joblib"))
    # Save scaler if we have one
    if scaler:
        torch.save(scaler, os.path.join(args.model_dir, "scaler.pt"))
    # Now test the prediction function
    print("Testing prediction function...")
    model_loaded = model_fn(args.model_dir)
    test_df = df_val.copy()
    print(f"Testing model with {len(test_df)} rows...")
    predictions_df = predict_fn(test_df, model_loaded)
    print(predictions_df.head())
    # Clean up temporary directories
    if os.path.exists("chemprop_outputs"):
        import shutil

        shutil.rmtree("chemprop_outputs")
        print("Removed chemprop_outputs directory.")
    print("Training and evaluation complete!")
