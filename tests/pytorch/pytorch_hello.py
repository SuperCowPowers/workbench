import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

# Import pytorch-tabular components
# Set this before importing any PyTorch-related modules to fix weight loading issues
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import pytest  # noqa: E402

pytest.skip("skipping this entire module for now", allow_module_level=True)
from pytorch_tabular import TabularModel  # noqa: E402
from pytorch_tabular.models import TabNetModelConfig, CategoryEmbeddingModelConfig  # noqa: E402
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig  # noqa: E402


def process_predictions(result, target_name, label_encoder=None, model_type="classifier"):
    """
    Process the prediction results from pytorch-tabular
    """
    df_result = result.copy()

    if model_type == "classifier":
        # Get predictions from {target}_prediction column
        pred_col = f"{target_name}_prediction"
        preds = df_result[pred_col].values

        # Get probability columns - they should be {target}_0_probability, {target}_1_probability, etc.
        prob_cols = [
            col for col in df_result.columns if col.startswith(f"{target_name}_") and col.endswith("_probability")
        ]
        prob_cols.sort()  # Ensure consistent ordering

        if prob_cols and label_encoder:
            # Create a list of probabilities for each row
            probs = df_result[prob_cols].values
            df_result = df_result.copy()
            df_result["pred_proba"] = [p.tolist() for p in probs]

            # Also create named columns for each class
            for i, class_name in enumerate(label_encoder.classes_):
                if i < len(prob_cols):
                    df_result[f"{class_name}_proba"] = df_result[prob_cols[i]]

        # Decode predictions if we have a label encoder
        if label_encoder:
            preds_decoded = label_encoder.inverse_transform(preds.astype(int))
        else:
            preds_decoded = preds

        return df_result, preds_decoded

    else:  # regression
        # For regression, look for {target}_prediction
        pred_col = f"{target_name}_prediction"
        if pred_col in df_result.columns:
            preds = df_result[pred_col].values
        else:
            # Fallback to target column name
            preds = df_result[target_name].values

        return df_result, preds


def create_sample_data(model_type="classifier", n_samples=5000):
    """
    Create sample training and validation data
    """
    np.random.seed(42)

    # Create sample features
    data = {
        "feature1": np.random.randn(n_samples),
        "feature2": np.random.randn(n_samples),
        "feature3": np.random.choice(["A", "B", "C"], n_samples),
        "feature4": np.random.randint(0, 100, n_samples).astype(float),
    }

    if model_type == "classifier":
        # Create binary classification target
        data["target"] = np.random.choice(["Class1", "Class2"], n_samples)
    else:
        # Create regression target
        data["target"] = 2 * data["feature1"] + 1.5 * data["feature2"] + np.random.randn(n_samples) * 0.1

    df = pd.DataFrame(data)

    # Split into train/validation
    split_idx = int(0.8 * len(df))
    df_train = df[:split_idx].copy()
    df_val = df[split_idx:].copy()

    return df_train, df_val


def test_model_training(model_type="classifier"):
    """
    Test the model training pipeline
    """
    print(f"Testing {model_type} model...")

    # Create sample data
    df_train, df_val = create_sample_data(model_type)
    target = "target"

    # Configure data settings - minimal required config
    continuous_cols = ["feature1", "feature2", "feature4"]
    categorical_cols = ["feature3"]

    data_config = DataConfig(
        target=[target],
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )

    # Use default configs - only override what we need
    optimizer_config = OptimizerConfig()
    trainer_config = TrainerConfig(
        auto_lr_find=False,  # Disable to avoid issues with small dataset
        max_epochs=10,  # Just set epochs, use defaults for everything else
    )

    # Choose model configuration based on model type
    if model_type == "classifier":
        task = "classification"
        model_config = TabNetModelConfig(task=task)
        # Encode the target column - create new dataframes to avoid chained assignment
        label_encoder = LabelEncoder()

        # Create completely fresh copies and assign the target column properly
        df_train_encoded = df_train.copy()
        df_val_encoded = df_val.copy()

        # Use proper assignment to avoid pandas warnings
        target_train_encoded = label_encoder.fit_transform(df_train[target])
        target_val_encoded = label_encoder.transform(df_val[target])

        df_train_encoded[target] = target_train_encoded
        df_val_encoded[target] = target_val_encoded
    else:
        task = "regression"
        model_config = CategoryEmbeddingModelConfig(task=task)
        label_encoder = None
        df_train_encoded = df_train
        df_val_encoded = df_val
    # Create and train the TabularModel
    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    # Train the model
    print("Training model...")
    tabular_model.fit(train=df_train_encoded, validation=df_val_encoded)
    # Make Predictions on the Validation Set
    print("Making Predictions on Validation Set...")
    result = tabular_model.predict(df_val_encoded)

    print("Available columns in result:", result.columns.tolist())
    # Process predictions
    df_result, preds = process_predictions(result, target, label_encoder, model_type)

    # Get actual target values for comparison
    if model_type == "classifier":
        y_validate = label_encoder.inverse_transform(df_val_encoded[target])
    else:
        y_validate = df_val_encoded[target].values
    # Add the decoded predictions to the result dataframe
    df_result = df_result.copy()
    df_result["prediction"] = preds

    # Define output columns based on what we actually have
    output_columns = []

    # Always include target and prediction
    if target in df_result.columns:
        output_columns.append(target)
    output_columns.append("prediction")

    # Add probability columns if they exist
    if model_type == "classifier":
        # Add the pytorch-tabular native probability columns
        prob_cols = [col for col in df_result.columns if col.startswith(f"{target}_") and col.endswith("_probability")]
        output_columns.extend(sorted(prob_cols))

        # Add our custom named probability columns
        class_prob_cols = [col for col in df_result.columns if col.endswith("_proba") and not col == "pred_proba"]
        output_columns.extend(sorted(class_prob_cols))

        # Add the list version if it exists
        if "pred_proba" in df_result.columns:
            output_columns.append("pred_proba")

    # Report Performance Metrics
    if model_type == "classifier":
        label_names = label_encoder.classes_
        scores = precision_recall_fscore_support(y_validate, preds, average=None, labels=label_names)
        score_df = pd.DataFrame(
            {
                target: label_names,
                "precision": scores[0],
                "recall": scores[1],
                "fscore": scores[2],
                "support": scores[3],
            }
        )
        print("\nClassification Metrics:")
        metrics = ["precision", "recall", "fscore", "support"]
        for t in label_names:
            for m in metrics:
                value = score_df.loc[score_df[target] == t, m].iloc[0]
                print(f"  {t} {m}: {value:.3f}")
        # Confusion matrix
        conf_mtx = confusion_matrix(y_validate, preds, labels=label_names)
        print("\nConfusion Matrix:")
        for i, row_name in enumerate(label_names):
            for j, col_name in enumerate(label_names):
                value = conf_mtx[i, j]
                print(f"  {row_name} -> {col_name}: {value}")
    else:
        # Regression metrics
        rmse = root_mean_squared_error(y_validate, preds)
        mae = mean_absolute_error(y_validate, preds)
        r2 = r2_score(y_validate, preds)
        print("\nRegression Metrics:")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  R2: {r2:.3f}")
        print(f"  NumRows: {len(df_result)}")


if __name__ == "__main__":
    # Test both classifier and regression
    print("=" * 50)
    test_model_training("classifier")

    print("\n" + "=" * 50)
    test_model_training("regression")
