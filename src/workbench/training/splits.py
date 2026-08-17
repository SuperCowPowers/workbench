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
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina

    from workbench.utils.chem_utils.fingerprints import similarity_fingerprints

    fps, valid_indices = similarity_fingerprints(smiles_list)
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


def analog_holdout_split(
    df: pd.DataFrame,
    target_columns: str | list[str],
    smiles_column: str | None = None,
    n_hits: int = 25,
    analogs_per_hit: int = 10,
    min_similarity: float = 0.4,
) -> np.ndarray:
    """Hold out close analogs of the most potent compounds, keeping the hits in training.

    Mimics a test set built by hit expansion: take the top hits per target, then hold out
    each hit's nearest neighbors by Tanimoto similarity. The result is dense clusters of
    near-neighbors around potent compounds rather than a diverse draw, so a model is
    scored on resolving small structural changes within a series -- the regime a random
    or scaffold split flatters.

    **The hits themselves stay in training**, which is how hit-expansion test sets are
    actually built: the hits are the already-measured compounds that got expanded, and
    the purchased analogs are what gets assayed blind. Holding the hits out instead puts
    the N most potent compounds in the eval set, which inflates its mean against training
    and shows up as a systematic under-prediction that is an artifact of the split.

    Unlike the grouping strategies behind `get_split_indices`, this is target-aware and
    yields one deliberate holdout, not a set of interchangeable folds.

    Args:
        df: DataFrame containing the data
        target_columns: Target column(s). Hits are taken per target and the holdouts unioned,
            so every target contributes its own potent series. A hit for any target is kept
            out of the holdout for all of them.
        smiles_column: Column containing SMILES. If None, auto-detects 'smiles' (case-insensitive)
        n_hits: Number of top-potency hits to expand per target
        analogs_per_hit: Neighbors to pull per hit
        min_similarity: Minimum Tanimoto similarity for a neighbor to count as an analog.
            Hits with fewer qualifying neighbors contribute a smaller cluster.

    Returns:
        Boolean mask over the rows of df, True for held-out rows.

    Raises:
        ValueError: If no SMILES column is found or no molecule can be parsed.

    Example:
        >>> holdout = analog_holdout_split(df, target_columns=["cyp3a4_pic50", "cyp2d6_pic50"])
        >>> train_df, eval_df = df[~holdout], df[holdout]
    """
    from rdkit import DataStructs

    from workbench.utils.chem_utils.fingerprints import similarity_fingerprints

    if isinstance(target_columns, str):
        target_columns = [target_columns]

    if smiles_column is None:
        smiles_column = _find_smiles_column(df.columns.tolist())
    if smiles_column is None or smiles_column not in df.columns:
        raise ValueError("analog_holdout_split needs a SMILES column; none found")

    missing = [c for c in target_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Target column(s) not found: {missing}")

    fps, valid_indices = similarity_fingerprints(df[smiles_column].tolist())
    if not fps:
        raise ValueError("No valid molecules found for the analog holdout")
    fp_position = {row: i for i, row in enumerate(valid_indices)}

    # Collect every target's hits before expanding, so a hit for one target is never
    # picked up as another target's analog -- measured hits belong in training.
    hits_by_target = {}
    for target in target_columns:
        values = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        ranked = [p for p in np.argsort(-values, kind="stable") if not np.isnan(values[p]) and p in fp_position]
        hits_by_target[target] = ranked[:n_hits]
    all_hits = {hit for hits in hits_by_target.values() for hit in hits}

    holdout = np.zeros(len(df), dtype=bool)
    for hits in hits_by_target.values():
        for hit in hits:
            sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[fp_position[hit]], fps))
            picked = 0
            for pos in np.argsort(-sims, kind="stable"):
                if sims[pos] < min_similarity or picked >= analogs_per_hit:
                    break
                row = valid_indices[pos]
                if row in all_hits:
                    continue
                holdout[row] = True
                picked += 1

    print(
        f"Analog holdout: {holdout.sum():,} of {len(df):,} rows "
        f"({holdout.sum() / len(df):.1%}) — analogs of {len(all_hits):,} hits "
        f"across {len(target_columns)} target(s), hits kept in training"
    )
    return holdout


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
