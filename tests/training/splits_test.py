"""Fast unit tests for ``workbench.training.splits``.

Covers the fold-partition contract every template and the HPO runner depend on:
grouped rows never straddle a boundary, every row is validated exactly once, a seed
reproduces a partition, and a different seed produces a different one. RDKit-backed
group assignment (scaffold/butina) is exercised through ``get_split_indices``; the
partition logic itself is tested directly on synthetic groups.
"""

import numpy as np
import pandas as pd
import pytest

# Workbench Imports
from workbench.training.splits import get_split_indices, group_folds

N_SPLITS = 5

# A scaffold-like group distribution: a heavy head carrying most rows, a long singleton
# tail. The head is what makes group assignment hard to randomize — it dominates both fold
# balance and how much two partitions differ.
HEAD_SIZES = [110, 95, 80, 70, 60, 55, 50, 45, 40, 35]
MID_SIZES = [12, 9, 7, 6, 5, 4, 3]
N_SINGLETONS = 200


def scaffold_like_groups() -> np.ndarray:
    """Per-row group labels with a heavy head and a singleton tail."""
    sizes = HEAD_SIZES + MID_SIZES + [1] * N_SINGLETONS
    return np.concatenate([[i] * size for i, size in enumerate(sizes)])


def fold_of_row(folds, n_rows: int) -> np.ndarray:
    """Flatten ``[(train_idx, val_idx), ...]`` to one fold label per row."""
    labels = np.empty(n_rows, dtype=int)
    for fold_idx, (_, val_idx) in enumerate(folds):
        labels[val_idx] = fold_idx
    return labels


def test_every_row_is_validated_exactly_once():
    """The val folds partition the rows — no row scored twice, none dropped."""
    groups = scaffold_like_groups()
    val_idx = np.concatenate([val for _, val in group_folds(groups, N_SPLITS, random_state=42)])
    assert sorted(val_idx) == list(range(len(groups)))


def test_train_and_val_are_disjoint_and_complete():
    """Each fold splits every row into exactly one of train or val."""
    groups = scaffold_like_groups()
    for train_idx, val_idx in group_folds(groups, N_SPLITS, random_state=42):
        assert set(train_idx).isdisjoint(val_idx)
        assert len(train_idx) + len(val_idx) == len(groups)


def test_no_group_straddles_a_fold_boundary():
    """The whole point of grouped folds: a scaffold in train is never also in val."""
    groups = scaffold_like_groups()
    for train_idx, val_idx in group_folds(groups, N_SPLITS, random_state=42):
        assert set(groups[train_idx]).isdisjoint(groups[val_idx])


def test_same_seed_reproduces_the_partition():
    """Determinism — a re-run of the same training job builds the same folds."""
    groups = scaffold_like_groups()
    first = group_folds(groups, N_SPLITS, random_state=42)
    second = group_folds(groups, N_SPLITS, random_state=42)
    for (a_train, a_val), (b_train, b_val) in zip(first, second):
        assert np.array_equal(a_train, b_train)
        assert np.array_equal(a_val, b_val)


def test_a_different_seed_moves_the_heavy_groups():
    """Changing the seed has to actually move the partition, heavy groups included.

    Checking the heavy head specifically: a size-ordered assignment pins the largest groups
    to the same fold at every seed, so most rows stay put and a "different" split is barely
    different — which would quietly weaken any comparison across seeds.
    """
    groups = scaffold_like_groups()
    head = np.isin(groups, range(len(HEAD_SIZES)))
    a = fold_of_row(group_folds(groups, N_SPLITS, random_state=42), len(groups))
    b = fold_of_row(group_folds(groups, N_SPLITS, random_state=1042), len(groups))
    assert not np.array_equal(a[head], b[head])


def test_folds_stay_near_even():
    """Fold sizes matter: callers macro-average metrics across folds, so a lopsided
    partition weights the small folds up."""
    groups = scaffold_like_groups()
    sizes = [len(val) for _, val in group_folds(groups, N_SPLITS, random_state=42)]
    even = len(groups) / N_SPLITS
    assert max(sizes) < 1.35 * even
    assert min(sizes) > 0.65 * even


def test_fewer_groups_than_folds_raises():
    """An unsatisfiable request fails loudly rather than emitting empty folds."""
    with pytest.raises(ValueError, match="number of groups"):
        group_folds(np.array([0, 0, 1, 1]), N_SPLITS, random_state=42)


# Molecule frame with enough distinct Bemis-Murcko scaffolds to fill five folds.
SMILES = [
    "CCO",
    "c1ccccc1",
    "CCN",
    "c1ccccc1C",
    "CCC",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "C1CCCCC1",
    "c1ccc(cc1)C(=O)O",
    "C1CCNCC1",
    "c1cnc2ccccc2c1",
    "C1COCCN1",
    "c1ccc2[nH]ccc2c1",
    "CC(C)Cc1ccccc1",
]


@pytest.fixture
def molecule_df() -> pd.DataFrame:
    return pd.DataFrame({"smiles": SMILES * 10, "y": range(10 * len(SMILES))})


@pytest.mark.parametrize("strategy", ["random", "scaffold", "butina"])
def test_strategies_partition_the_frame(molecule_df, strategy):
    """Every strategy returns n_splits folds whose val sets partition the rows."""
    folds = get_split_indices(molecule_df, n_splits=N_SPLITS, strategy=strategy)
    assert len(folds) == N_SPLITS
    val_idx = np.concatenate([val for _, val in folds])
    assert sorted(val_idx) == list(range(len(molecule_df)))


@pytest.mark.parametrize("strategy", ["scaffold", "butina"])
def test_molecule_strategies_keep_identical_smiles_together(molecule_df, strategy):
    """Same molecule, same fold — near-duplicates leaking across the boundary is the
    failure these strategies exist to prevent."""
    folds = get_split_indices(molecule_df, n_splits=N_SPLITS, strategy=strategy)
    smiles = molecule_df["smiles"].to_numpy()
    for train_idx, val_idx in folds:
        assert set(smiles[train_idx]).isdisjoint(smiles[val_idx])


def test_single_split_returns_one_train_val_pair(molecule_df):
    """n_splits=1 is a single holdout split, not a fold partition."""
    folds = get_split_indices(molecule_df, n_splits=1, strategy="scaffold", test_size=0.2)
    assert len(folds) == 1
    train_idx, val_idx = folds[0]
    assert len(train_idx) + len(val_idx) == len(molecule_df)
    assert set(train_idx).isdisjoint(val_idx)


def test_missing_smiles_column_falls_back_to_random():
    """A non-molecular frame asking for scaffold folds degrades rather than raising."""
    df = pd.DataFrame({"feature": range(100), "y": range(100)})
    folds = get_split_indices(df, n_splits=N_SPLITS, strategy="scaffold")
    assert len(folds) == N_SPLITS
    assert sorted(np.concatenate([val for _, val in folds])) == list(range(len(df)))


def test_unknown_strategy_raises(molecule_df):
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_split_indices(molecule_df, n_splits=N_SPLITS, strategy="nope")
