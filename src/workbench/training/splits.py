"""Dataset splitting for model training — random, scaffold, and Butina strategies.

Training-only (per the :mod:`workbench.training` contract); templates import this
**only inside their ``__main__``** (deferred). :func:`get_split_indices` is the one
fold interface every model template and the HPO runner build their ensembles from, so
train/validation partitioning behaves identically across frameworks.

Scaffold (Bemis-Murcko) and Butina (Morgan-fingerprint clustering) strategies group
molecules so near-duplicates never straddle a train/validation boundary; random is the
non-molecular fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_scaffold(smiles: str) -> str:
    """Extract Bemis-Murcko scaffold from a SMILES string.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        SMILES string of the scaffold, or empty string if molecule is invalid
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # RDKit raises TypeError on non-strings (e.g. NaN floats from CSV), so guard first
    if not isinstance(smiles, str) or not smiles:
        return ""
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
        # Guard non-strings (e.g. NaN) before RDKit — it raises TypeError on floats
        if not isinstance(smi, str) or not smi:
            continue
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
    valid_set = set(valid_indices)
    next_cluster = len(clusters)
    for i in range(len(smiles_list)):
        if i not in valid_set:
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

    if n_splits == 1:
        # Single split: use GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return list(splitter.split(df, groups=groups))
    return group_folds(groups, n_splits=n_splits, random_state=random_state)


def group_folds(groups: np.ndarray, n_splits: int, random_state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition grouped rows into ``n_splits`` folds, keeping every group intact.

    Groups are visited in a seed-dependent random order and each lands in whichever fold
    is currently lightest. The random order is what makes the partition depend on
    ``random_state`` — including for the largest groups, which carry most of the rows and
    so dominate how much two partitions differ. The greedy fill keeps fold sizes near-even,
    which matters when a caller macro-averages a metric across folds.

    Args:
        groups: per-row group label; rows sharing a label never straddle a fold boundary.
        n_splits: number of folds.
        random_state: seed for the group ordering.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    unique_groups, counts = np.unique(groups, return_counts=True)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Cannot have n_splits={n_splits} greater than the number of groups ({len(unique_groups)})")

    load = np.zeros(n_splits, dtype=np.int64)
    group_to_fold = {}
    for i in np.random.default_rng(random_state).permutation(len(unique_groups)):
        fold = int(np.argmin(load))
        group_to_fold[unique_groups[i]] = fold
        load[fold] += counts[i]

    fold_of_row = np.array([group_to_fold[g] for g in groups])
    indices = np.arange(len(groups))
    return [(indices[fold_of_row != f], indices[fold_of_row == f]) for f in range(n_splits)]
