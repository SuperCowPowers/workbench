"""PyTorch utilities for tabular data modeling.

Provides a lightweight TabularMLP model with categorical embeddings and
training utilities for use in Workbench model scripts.
"""

import json
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureScaler:
    """Standard scaler for continuous features (zero mean, unit variance)."""

    def __init__(self):
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.feature_names: Optional[list[str]] = None

    def fit(self, df: pd.DataFrame, continuous_cols: list[str]) -> "FeatureScaler":
        """Fit the scaler on training data."""
        self.feature_names = continuous_cols
        data = df[continuous_cols].values.astype(np.float32)
        self.means = np.nanmean(data, axis=0)
        self.stds = np.nanstd(data, axis=0)
        # Avoid division by zero for constant features
        self.stds[self.stds == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted parameters."""
        data = df[self.feature_names].values.astype(np.float32)
        # Fill NaN with mean before scaling
        for i, mean in enumerate(self.means):
            data[np.isnan(data[:, i]), i] = mean
        return (data - self.means) / self.stds

    def fit_transform(self, df: pd.DataFrame, continuous_cols: list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, continuous_cols)
        return self.transform(df)

    def save(self, path: str) -> None:
        """Save scaler parameters."""
        joblib.dump(
            {
                "means": self.means.tolist(),
                "stds": self.stds.tolist(),
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load scaler from saved parameters."""
        data = joblib.load(path)
        scaler = cls()
        scaler.means = np.array(data["means"], dtype=np.float32)
        scaler.stds = np.array(data["stds"], dtype=np.float32)
        scaler.feature_names = data["feature_names"]
        return scaler


class TabularMLP(nn.Module):
    """Feedforward neural network for tabular data with optional categorical embeddings.

    Args:
        n_continuous: Number of continuous input features
        categorical_cardinalities: List of cardinalities for each categorical feature
        embedding_dims: List of embedding dimensions for each categorical feature
        hidden_layers: List of hidden layer sizes (e.g., [256, 128, 64])
        n_outputs: Number of output units
        task: "regression" or "classification"
        dropout: Dropout rate
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_cardinalities: list[int],
        embedding_dims: list[int],
        hidden_layers: list[int],
        n_outputs: int,
        task: str = "regression",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.task = task
        self.n_continuous = n_continuous
        self.categorical_cardinalities = categorical_cardinalities

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n_cats, emb_dim) for n_cats, emb_dim in zip(categorical_cardinalities, embedding_dims)]
        )

        # Calculate input dimension
        total_emb_dim = sum(embedding_dims)
        input_dim = n_continuous + total_emb_dim

        # Build MLP layers
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(input_dim, n_outputs)

    def forward(self, x_cont: torch.Tensor, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x_cont: Continuous features tensor of shape (batch, n_continuous)
            x_cat: Categorical features tensor of shape (batch, n_categoricals), optional

        Returns:
            Output tensor of shape (batch, n_outputs)
        """
        # Embed categorical features and concatenate with continuous
        if x_cat is not None and len(self.embeddings) > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_cont] + embs, dim=1)
        else:
            x = x_cont

        x = self.mlp(x)
        out = self.head(x)

        if self.task == "classification":
            out = torch.softmax(out, dim=1)

        return out


def compute_embedding_dims(cardinalities: list[int], max_dim: int = 50) -> list[int]:
    """Compute embedding dimensions using the rule of thumb: min(50, (n+1)//2)."""
    return [min(max_dim, (n + 1) // 2) for n in cardinalities]


def prepare_data(
    df: pd.DataFrame,
    continuous_cols: list[str],
    categorical_cols: list[str],
    target_col: Optional[str] = None,
    category_mappings: Optional[dict] = None,
    scaler: Optional[FeatureScaler] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], dict, Optional[FeatureScaler]]:
    """Prepare dataframe for model input.

    Args:
        df: Input dataframe
        continuous_cols: List of continuous feature column names
        categorical_cols: List of categorical feature column names
        target_col: Target column name (optional, for training)
        category_mappings: Existing category mappings (for inference)
        scaler: Existing FeatureScaler (for inference), or None to fit a new one

    Returns:
        Tuple of (x_cont, x_cat, y, category_mappings, scaler)
    """
    # Continuous features with standardization
    if scaler is None:
        scaler = FeatureScaler()
        cont_data = scaler.fit_transform(df, continuous_cols)
    else:
        cont_data = scaler.transform(df)
    x_cont = torch.tensor(cont_data, dtype=torch.float32)

    # Categorical features
    x_cat = None
    if categorical_cols:
        if category_mappings is None:
            category_mappings = {}
            for col in categorical_cols:
                unique_vals = df[col].unique().tolist()
                category_mappings[col] = {v: i for i, v in enumerate(unique_vals)}

        cat_indices = []
        for col in categorical_cols:
            mapping = category_mappings[col]
            # Map values to indices, use 0 for unknown categories
            indices = df[col].map(lambda x: mapping.get(x, 0)).values
            cat_indices.append(indices)

        x_cat = torch.tensor(np.column_stack(cat_indices), dtype=torch.long)

    # Target
    y = None
    if target_col is not None:
        y = torch.tensor(df[target_col].values, dtype=torch.float32)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

    return x_cont, x_cat, y, category_mappings, scaler


def create_model(
    n_continuous: int,
    categorical_cardinalities: list[int],
    hidden_layers: list[int],
    n_outputs: int,
    task: str = "regression",
    dropout: float = 0.1,
    use_batch_norm: bool = True,
) -> TabularMLP:
    """Create a TabularMLP model with appropriate embedding dimensions."""
    embedding_dims = compute_embedding_dims(categorical_cardinalities)
    return TabularMLP(
        n_continuous=n_continuous,
        categorical_cardinalities=categorical_cardinalities,
        embedding_dims=embedding_dims,
        hidden_layers=hidden_layers,
        n_outputs=n_outputs,
        task=task,
        dropout=dropout,
        use_batch_norm=use_batch_norm,
    )


def train_model(
    model: TabularMLP,
    train_x_cont: torch.Tensor,
    train_x_cat: Optional[torch.Tensor],
    train_y: torch.Tensor,
    val_x_cont: torch.Tensor,
    val_x_cat: Optional[torch.Tensor],
    val_y: torch.Tensor,
    task: str = "regression",
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    loss: str = "L1Loss",
    device: str = "cpu",
) -> tuple[TabularMLP, dict]:
    """Train the model with early stopping.

    Returns:
        Tuple of (trained model, training history dict)
    """
    model = model.to(device)

    # Create dataloaders
    if train_x_cat is not None:
        train_dataset = TensorDataset(train_x_cont, train_x_cat, train_y)
        val_dataset = TensorDataset(val_x_cont, val_x_cat, val_y)
    else:
        # Use dummy categorical tensor
        dummy_cat = torch.zeros(train_x_cont.shape[0], 0, dtype=torch.long)
        dummy_val_cat = torch.zeros(val_x_cont.shape[0], 0, dtype=torch.long)
        train_dataset = TensorDataset(train_x_cont, dummy_cat, train_y)
        val_dataset = TensorDataset(val_x_cont, dummy_val_cat, val_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss and optimizer
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        # Map loss name to PyTorch loss class
        loss_map = {
            "L1Loss": nn.L1Loss,
            "MSELoss": nn.MSELoss,
            "HuberLoss": nn.HuberLoss,
            "SmoothL1Loss": nn.SmoothL1Loss,
        }
        if loss not in loss_map:
            raise ValueError(f"Unknown loss '{loss}'. Supported: {list(loss_map.keys())}")
        criterion = loss_map[loss]()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with early stopping
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            x_cont, x_cat, y = [b.to(device) for b in batch]
            x_cat = x_cat if x_cat.shape[1] > 0 else None

            optimizer.zero_grad()
            out = model(x_cont, x_cat)

            if task == "classification":
                loss = criterion(out, y.squeeze().long())
            else:
                loss = criterion(out, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x_cont, x_cat, y = [b.to(device) for b in batch]
                x_cat = x_cat if x_cat.shape[1] > 0 else None
                out = model(x_cont, x_cat)

                if task == "classification":
                    loss = criterion(out, y.squeeze().long())
                else:
                    loss = criterion(out, y)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    model = model.to("cpu")
    return model, history


def predict(
    model: TabularMLP,
    x_cont: torch.Tensor,
    x_cat: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference with the model."""
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        x_cont = x_cont.to(device)
        if x_cat is not None:
            x_cat = x_cat.to(device)
        out = model(x_cont, x_cat)

    return out.cpu().numpy()


def save_model(model: TabularMLP, path: str, model_config: dict) -> None:
    """Save model weights and configuration."""
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)


def load_model(path: str, device: str = "cpu") -> TabularMLP:
    """Load model from saved weights and configuration."""
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)

    model = create_model(
        n_continuous=config["n_continuous"],
        categorical_cardinalities=config["categorical_cardinalities"],
        hidden_layers=config["hidden_layers"],
        n_outputs=config["n_outputs"],
        task=config["task"],
        dropout=config.get("dropout", 0.1),
        use_batch_norm=config.get("use_batch_norm", True),
    )

    state_dict = torch.load(os.path.join(path, "model.pt"), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model
