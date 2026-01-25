"""Shared utility functions for model training scripts (templates).

These functions are used across multiple model templates (XGBoost, PyTorch, ChemProp)
to reduce code duplication and ensure consistent behavior.
"""

from io import StringIO
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scipy.stats import spearmanr


def check_dataframe(df: pd.DataFrame, df_name: str) -> None:
    """Check if the provided dataframe is empty and raise an exception if it is.

    Args:
        df: DataFrame to check
        df_name: Name of the DataFrame (for error message)

    Raises:
        ValueError: If the DataFrame is empty
    """
    if df.empty:
        msg = f"*** The training data {df_name} has 0 rows! ***STOPPING***"
        print(msg)
        raise ValueError(msg)


def expand_proba_column(df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Expands a column containing a list of probabilities into separate columns.

    Handles None values for rows where predictions couldn't be made.

    Args:
        df: DataFrame containing a "pred_proba" column
        class_labels: List of class labels

    Returns:
        DataFrame with the "pred_proba" expanded into separate columns (e.g., "class1_proba")

    Raises:
        ValueError: If DataFrame does not contain a "pred_proba" column
    """
    proba_column = "pred_proba"
    if proba_column not in df.columns:
        raise ValueError('DataFrame does not contain a "pred_proba" column')

    proba_splits = [f"{label}_proba" for label in class_labels]
    n_classes = len(class_labels)

    # Handle None values by replacing with list of NaNs
    proba_values = []
    for val in df[proba_column]:
        if val is None:
            proba_values.append([np.nan] * n_classes)
        else:
            proba_values.append(val)

    proba_df = pd.DataFrame(proba_values, columns=proba_splits)

    # Drop any existing proba columns and reset index for concat
    df = df.drop(columns=[proba_column] + proba_splits, errors="ignore")
    df = df.reset_index(drop=True)
    df = pd.concat([df, proba_df], axis=1)
    return df


def match_features_case_insensitive(df: pd.DataFrame, model_features: list[str]) -> pd.DataFrame:
    """Matches and renames DataFrame columns to match model feature names (case-insensitive).

    Prioritizes exact matches, then case-insensitive matches.

    Args:
        df: Input DataFrame
        model_features: List of feature names expected by the model

    Returns:
        DataFrame with columns renamed to match model features

    Raises:
        ValueError: If any model features cannot be matched
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


def convert_categorical_types(
    df: pd.DataFrame, features: list[str], category_mappings: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Converts appropriate columns to categorical type with consistent mappings.

    In training mode (category_mappings is None or empty), detects object/string columns
    with <20 unique values and converts them to categorical.
    In inference mode (category_mappings provided), applies the stored mappings.

    Args:
        df: The DataFrame to process
        features: List of feature names to consider for conversion
        category_mappings: Existing category mappings. If None or empty, training mode.
                          If populated, inference mode.

    Returns:
        Tuple of (processed DataFrame, category mappings dictionary)
    """
    if category_mappings is None:
        category_mappings = {}

    # Training mode
    if not category_mappings:
        for col in df.select_dtypes(include=["object", "string"]):
            if col in features and df[col].nunique() < 20:
                print(f"Training mode: Converting {col} to category")
                df[col] = df[col].astype("category")
                category_mappings[col] = df[col].cat.categories.tolist()

    # Inference mode
    else:
        for col, categories in category_mappings.items():
            if col in df.columns:
                print(f"Inference mode: Applying categorical mapping for {col}")
                df[col] = pd.Categorical(df[col], categories=categories)

    return df, category_mappings


def decompress_features(
    df: pd.DataFrame, features: list[str], compressed_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Decompress compressed features (bitstrings or count vectors) into individual columns.

    Supports two formats (auto-detected):
        - Bitstrings: "10110010..." → individual uint8 columns (0 or 1)
        - Count vectors: "0,3,0,1,5,..." → individual uint8 columns (0-255)

    Args:
        df: The features DataFrame
        features: Full list of feature names
        compressed_features: List of feature names to decompress

    Returns:
        Tuple of (DataFrame with decompressed features, updated feature list)
    """
    # Check for any missing values in the required features
    missing_counts = df[features].isna().sum()
    if missing_counts.any():
        missing_features = missing_counts[missing_counts > 0]
        print(
            f"WARNING: Found missing values in features: {missing_features.to_dict()}. "
            "WARNING: You might want to remove/replace all NaN values before processing."
        )

    # Make a copy to avoid mutating the original list
    decompressed_features = features.copy()

    for feature in compressed_features:
        if (feature not in df.columns) or (feature not in decompressed_features):
            print(f"Feature '{feature}' not in the features list, skipping decompression.")
            continue

        # Remove the feature from the list to avoid duplication
        decompressed_features.remove(feature)

        # Auto-detect format and parse: comma-separated counts or bitstring
        sample = str(df[feature].dropna().iloc[0]) if not df[feature].dropna().empty else ""
        parse_fn = (lambda s: list(map(int, s.split(",")))) if "," in sample else list
        feature_matrix = np.array([parse_fn(s) for s in df[feature]], dtype=np.uint8)

        # Create new columns with prefix from feature name
        prefix = feature[:3]
        new_col_names = [f"{prefix}_{i}" for i in range(feature_matrix.shape[1])]
        new_df = pd.DataFrame(feature_matrix, columns=new_col_names, index=df.index)

        # Update features list and dataframe
        decompressed_features.extend(new_col_names)
        df = df.drop(columns=[feature])
        df = pd.concat([df, new_df], axis=1)

    return df, decompressed_features


def input_fn(input_data, content_type: str) -> pd.DataFrame:
    """Parse input data and return a DataFrame.

    Args:
        input_data: Raw input data (bytes or string)
        content_type: MIME type of the input data

    Returns:
        Parsed DataFrame

    Raises:
        ValueError: If input is empty or content_type is not supported
    """
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df: pd.DataFrame, accept_type: str) -> tuple[str, str]:
    """Convert output DataFrame to requested format.

    Args:
        output_df: DataFrame to convert
        accept_type: Requested MIME type

    Returns:
        Tuple of (formatted output string, MIME type)

    Raises:
        RuntimeError: If accept_type is not supported
    """
    if "text/csv" in accept_type:
        csv_output = output_df.fillna("N/A").to_csv(index=False)
        return csv_output, "text/csv"
    elif "application/json" in accept_type:
        return output_df.to_json(orient="records"), "application/json"
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


def cap_std_outliers(std_array: np.ndarray) -> np.ndarray:
    """Cap extreme outliers in prediction_std using IQR method.

    Uses the standard IQR fence (Q3 + 1.5*IQR) to cap extreme values.
    This prevents unreasonably large std values while preserving the
    relative ordering and keeping meaningful high-uncertainty signals.

    Args:
        std_array: Array of standard deviations (n_samples,) or (n_samples, n_targets)

    Returns:
        Array with outliers capped at the upper fence
    """
    if std_array.ndim == 1:
        std_array = std_array.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    capped = std_array.copy()
    for col in range(capped.shape[1]):
        col_data = capped[:, col]
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        capped[:, col] = np.minimum(col_data, upper_bound)

    return capped.squeeze() if squeeze else capped


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth target values
        y_pred: Predicted values

    Returns:
        Dictionary with keys: rmse, mae, medae, r2, spearmanr, support
    """
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "spearmanr": spearmanr(y_true, y_pred).correlation,
        "support": len(y_true),
    }


def print_regression_metrics(metrics: dict[str, float]) -> None:
    """Print regression metrics in the format expected by SageMaker metric definitions.

    Args:
        metrics: Dictionary of metric name -> value
    """
    print(f"rmse: {metrics['rmse']:.3f}")
    print(f"mae: {metrics['mae']:.3f}")
    print(f"medae: {metrics['medae']:.3f}")
    print(f"r2: {metrics['r2']:.3f}")
    print(f"spearmanr: {metrics['spearmanr']:.3f}")
    print(f"support: {metrics['support']}")


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], target_col: str
) -> pd.DataFrame:
    """Compute per-class classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class label names
        target_col: Name of the target column (for DataFrame output)

    Returns:
        DataFrame with columns: target_col, precision, recall, f1, support
    """
    scores = precision_recall_fscore_support(y_true, y_pred, average=None, labels=label_names)
    return pd.DataFrame(
        {
            target_col: label_names,
            "precision": scores[0],
            "recall": scores[1],
            "f1": scores[2],
            "support": scores[3],
        }
    )


def print_classification_metrics(score_df: pd.DataFrame, target_col: str, label_names: list[str]) -> None:
    """Print per-class classification metrics in the format expected by SageMaker.

    Args:
        score_df: DataFrame from compute_classification_metrics
        target_col: Name of the target column
        label_names: List of class label names
    """
    metrics = ["precision", "recall", "f1", "support"]
    for t in label_names:
        for m in metrics:
            value = score_df.loc[score_df[target_col] == t, m].iloc[0]
            print(f"Metrics:{t}:{m} {value}")


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> None:
    """Print confusion matrix in the format expected by SageMaker.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label_names: List of class label names
    """
    conf_mtx = confusion_matrix(y_true, y_pred, labels=label_names)
    for i, row_name in enumerate(label_names):
        for j, col_name in enumerate(label_names):
            value = conf_mtx[i, j]
            print(f"ConfusionMatrix:{row_name}:{col_name} {value}")


# =============================================================================
# Dataset Splitting Utilities for Molecular Data
# =============================================================================
def get_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from a SMILES string.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        SMILES string of the scaffold, or empty string if molecule is invalid
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return ""


def get_scaffold_groups(smiles_list: list[str]) -> np.ndarray:
    """Assign each molecule to a scaffold group.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Array of group indices (same scaffold = same group)
    """
    scaffold_to_group = {}
    groups = []

    for smi in smiles_list:
        scaffold = get_scaffold(smi)
        if scaffold not in scaffold_to_group:
            scaffold_to_group[scaffold] = len(scaffold_to_group)
        groups.append(scaffold_to_group[scaffold])

    n_scaffolds = len(scaffold_to_group)
    print(f"Found {n_scaffolds} unique scaffolds from {len(smiles_list)} molecules")
    return np.array(groups)


def get_butina_clusters(smiles_list: list[str], cutoff: float = 0.4) -> np.ndarray:
    """Cluster molecules using Butina algorithm on Morgan fingerprints.

    Uses RDKit's Butina clustering with Tanimoto distance on Morgan fingerprints.
    This is Pat Walters' recommended approach for creating diverse train/test splits.

    Args:
        smiles_list: List of SMILES strings
        cutoff: Tanimoto distance cutoff for clustering (default 0.4)
               Lower values = more clusters = more similar molecules per cluster

    Returns:
        Array of cluster indices
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    from rdkit.ML.Cluster import Butina

    # Create Morgan fingerprint generator
    fp_gen = GetMorganGenerator(radius=2, fpSize=2048)

    # Generate Morgan fingerprints
    fps = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = fp_gen.GetFingerprint(mol)
            fps.append(fp)
            valid_indices.append(i)

    if len(fps) == 0:
        raise ValueError("No valid molecules found for clustering")

    # Compute distance matrix (upper triangle only for efficiency)
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])

    # Butina clustering
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)

    # Map back to original indices
    cluster_labels = np.zeros(len(smiles_list), dtype=int)
    for cluster_idx, cluster in enumerate(clusters):
        for mol_idx in cluster:
            original_idx = valid_indices[mol_idx]
            cluster_labels[original_idx] = cluster_idx

    # Assign invalid molecules to their own clusters
    next_cluster = len(clusters)
    for i in range(len(smiles_list)):
        if i not in valid_indices:
            cluster_labels[i] = next_cluster
            next_cluster += 1

    n_clusters = len(set(cluster_labels))
    print(f"Butina clustering: {n_clusters} clusters from {len(smiles_list)} molecules (cutoff={cutoff})")
    return cluster_labels


def _find_smiles_column(columns: list[str]) -> str | None:
    """Find SMILES column (case-insensitive match for 'smiles').

    Args:
        columns: List of column names

    Returns:
        The matching column name, or None if not found
    """
    return next((c for c in columns if c.lower() == "smiles"), None)


def get_split_indices(
    df: pd.DataFrame,
    n_splits: int = 5,
    strategy: str = "random",
    smiles_column: str | None = None,
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    butina_cutoff: float = 0.4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Get train/validation split indices using various strategies.

    This is a unified interface for generating splits that can be used across
    all model templates (XGBoost, PyTorch, ChemProp).

    Args:
        df: DataFrame containing the data
        n_splits: Number of CV folds (1 = single train/val split)
        strategy: Split strategy - one of:
            - "random": Standard random split (default sklearn behavior)
            - "scaffold": Bemis-Murcko scaffold-based grouping
            - "butina": Morgan fingerprint clustering (recommended for ADMET)
        smiles_column: Column containing SMILES. If None, auto-detects 'smiles' (case-insensitive)
        target_column: Column containing target values (for stratification, optional)
        test_size: Fraction for validation set when n_splits=1 (default 0.2)
        random_state: Random seed for reproducibility
        butina_cutoff: Tanimoto distance cutoff for Butina clustering (default 0.4)

    Returns:
        List of (train_indices, val_indices) tuples

    Note:
        If scaffold/butina strategy is requested but no SMILES column is found,
        automatically falls back to random split with a warning message.

    Example:
        >>> folds = get_split_indices(df, n_splits=5, strategy="scaffold")
        >>> for train_idx, val_idx in folds:
        ...     X_train, X_val = df.iloc[train_idx], df.iloc[val_idx]
    """
    from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

    n_samples = len(df)

    # Random split (original behavior)
    if strategy == "random":
        if n_splits == 1:
            indices = np.arange(n_samples)
            train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
            return [(train_idx, val_idx)]
        else:
            if target_column and df[target_column].dtype in ["object", "category", "bool"]:
                kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                return list(kfold.split(df, df[target_column]))
            else:
                kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                return list(kfold.split(df))

    # Scaffold or Butina split requires SMILES - auto-detect if not provided
    if smiles_column is None:
        smiles_column = _find_smiles_column(df.columns.tolist())

    # Fall back to random split if no SMILES column available
    if smiles_column is None or smiles_column not in df.columns:
        print(f"No 'smiles' column found for strategy='{strategy}', falling back to random split")
        return get_split_indices(
            df,
            n_splits=n_splits,
            strategy="random",
            target_column=target_column,
            test_size=test_size,
            random_state=random_state,
        )

    smiles_list = df[smiles_column].tolist()

    # Get group assignments
    if strategy == "scaffold":
        groups = get_scaffold_groups(smiles_list)
    elif strategy == "butina":
        groups = get_butina_clusters(smiles_list, cutoff=butina_cutoff)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'random', 'scaffold', or 'butina'")

    # Generate splits using GroupKFold or GroupShuffleSplit
    if n_splits == 1:
        # Single split: use GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return list(splitter.split(df, groups=groups))
    else:
        # K-fold: use GroupKFold (ensures no group appears in both train and val)
        # Note: GroupKFold doesn't shuffle, so we shuffle group order first
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(random_state)
        shuffled_group_map = {g: i for i, g in enumerate(rng.permutation(unique_groups))}
        shuffled_groups = np.array([shuffled_group_map[g] for g in groups])

        gkf = GroupKFold(n_splits=n_splits)
        return list(gkf.split(df, groups=shuffled_groups))
