"""Unit tests for TargetLandscape (duplicates, cliffs, isolation).

Fingerprints are given as explicit bitstrings rather than derived from SMILES, so
distances are exact and the tests need neither RDKit nor AWS.
"""

import pandas as pd
import pytest

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity
from workbench.algorithms.dataframe.target_landscape import TargetLandscape


def landscape(ids, fingerprints, targets=None) -> TargetLandscape:
    """Build a TargetLandscape over an explicit fingerprint set."""
    data = {"id": ids, "fingerprint": fingerprints}
    if targets is not None:
        data["target"] = targets
    prox = FingerprintProximity(
        pd.DataFrame(data),
        id_column="id",
        fingerprint_column="fingerprint",
        target="target" if targets is not None else None,
    )
    return TargetLandscape(prox)


# Six distinct fingerprints, spread out enough that none are coincident.
DISTINCT = [
    "1111000000",
    "1110100000",
    "1100110000",
    "1000111000",
    "0000111100",
    "0000001111",
]


# ------------------------------------------------------------------
# duplicates()
# ------------------------------------------------------------------


def test_duplicates_groups_a_triple_once():
    """Three coincident rows form one group of size 3, not three pairs."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000"] * 3 + DISTINCT[3:],
        targets=[0.1, 3.0, 1.5, 2.0, -0.5, 0.7],
    )
    dups = land.duplicates()

    assert len(dups) == 3
    assert dups["group_id"].nunique() == 1
    assert set(dups["id"]) == {"a", "b", "c"}
    assert (dups["group_size"] == 3).all()


def test_duplicates_reports_spread_and_median():
    """group_spread is max-min and group_median the consensus, repeated per member."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000"] * 3 + DISTINCT[3:],
        targets=[0.1, 3.0, 1.5, 2.0, -0.5, 0.7],
    )
    dups = land.duplicates()

    assert dups["group_spread"].unique().tolist() == [pytest.approx(2.9)]
    assert dups["group_median"].unique().tolist() == [pytest.approx(1.5)]


def test_duplicates_separates_independent_groups():
    """Two unrelated coincident groups get distinct group_ids."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000", "1111000000", "0000001111", "0000001111"] + DISTINCT[1:3],
        targets=[1.0, 2.0, 5.0, 5.0, 3.0, 4.0],
    )
    dups = land.duplicates()

    assert len(dups) == 4
    assert dups["group_id"].nunique() == 2
    groups = {frozenset(g["id"]) for _, g in dups.groupby("group_id")}
    assert groups == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_duplicates_min_spread_filters_consistent_groups():
    """Groups that agree are dropped by min_spread; contradictory ones survive."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000", "1111000000", "0000001111", "0000001111"] + DISTINCT[1:3],
        targets=[1.0, 4.0, 5.0, 5.0, 3.0, 2.0],  # a/b spread 3.0, c/d spread 0.0
    )
    dups = land.duplicates(min_spread=1.0)

    assert set(dups["id"]) == {"a", "b"}


def test_duplicates_empty_when_none_coincident():
    """A set with no coincident rows returns an empty frame with the full schema."""
    land = landscape(ids=list("abcdef"), fingerprints=DISTINCT, targets=[1, 2, 3, 4, 5, 6])
    dups = land.duplicates()

    assert dups.empty
    assert list(dups.columns) == ["group_id", "id", "target", "group_spread", "group_median", "group_size"]


def test_coincident_rows_point_at_each_other_not_themselves():
    """A duplicate ties at distance 0 and can sort ahead of the row itself.

    Taking the second-nearest slot blindly would name the row its own neighbor and
    zero out its target difference, dropping it from every downstream analysis.
    """
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000", "1111000000"] + DISTINCT[1:5],
        targets=[0.0, 10.0, 1.0, 8.0, 2.0, 3.0],
    )
    land._ensure_metrics()
    metrics = land.prox.df.set_index("id")

    assert metrics.loc["a", "nn_id"] == "b"
    assert metrics.loc["b", "nn_id"] == "a"
    assert metrics.loc["a", "nn_target_diff"] == 10.0
    assert (metrics["nn_id"] != metrics.index).all()


def test_duplicates_without_target_omits_spread_columns():
    """With no target there's nothing to disagree about — just the grouping."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000"] * 3 + DISTINCT[3:],
    )
    dups = land.duplicates()

    assert list(dups.columns) == ["group_id", "id", "group_size"]
    assert len(dups) == 3


# ------------------------------------------------------------------
# cliffs() / target_gradients()
# ------------------------------------------------------------------


def test_cliffs_excludes_coincident_rows():
    """Coincident rows are duplicates, not cliffs — they never appear in cliffs()."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000", "1111000000"] + DISTINCT[1:5],
        targets=[0.0, 10.0, 1.0, 8.0, 2.0, 3.0],
    )
    cliffs = land.cliffs(top_percent=100.0)

    assert not cliffs.empty
    assert (cliffs["nn_distance"] > 0).all()
    assert not {"a", "b"} & set(cliffs["id"])


def test_only_coincident_returns_just_the_duplicates():
    """The opposite filter: only rows whose nearest neighbor is coincident."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "f"],
        fingerprints=["1111000000", "1111000000"] + DISTINCT[1:5],
        targets=[0.0, 10.0, 1.0, 8.0, 2.0, 3.0],
    )
    coincident = land.target_gradients(only_coincident=True)

    assert set(coincident["id"]) == {"a", "b"}
    assert (coincident["nn_distance"] == 0).all()


def test_cliff_score_is_invariant_to_target_units():
    """Normalizing by target range makes the score portable across endpoints."""
    ids, fps = list("abcdef"), DISTINCT
    targets = [0.0, 5.0, 1.0, 4.0, 2.0, 3.0]

    native = landscape(ids, fps, targets).cliffs(top_percent=100.0)
    scaled = landscape(ids, fps, [t * 1000 for t in targets]).cliffs(top_percent=100.0)

    assert native["cliff_score"].tolist() == pytest.approx(scaled["cliff_score"].tolist())


def test_target_gradients_rejects_conflicting_filters():
    """only_coincident and min_distance ask for opposite things."""
    land = landscape(list("abcdef"), DISTINCT, [1, 2, 3, 4, 5, 6])
    with pytest.raises(ValueError, match="mutually exclusive"):
        land.target_gradients(only_coincident=True, min_distance=1e-6)


def test_target_gradients_requires_a_target():
    """Gradients are meaningless without a target column."""
    land = landscape(list("abcdef"), DISTINCT)
    with pytest.raises(ValueError, match="requires a Proximity backend with `target` set"):
        land.target_gradients()


def test_cliffs_empty_when_every_neighbor_is_coincident():
    """Excluding coincident rows can leave nothing — that's an empty frame, not a crash."""
    land = landscape(
        ids=["a", "b", "c", "d"],
        fingerprints=["1111000000", "1111000000", "0000001111", "0000001111"],
        targets=[0.0, 9.0, 1.0, 8.0],
    )
    cliffs = land.cliffs()

    assert cliffs.empty
    assert "cliff_score" in cliffs.columns


# ------------------------------------------------------------------
# isolated() / proximity_stats()
# ------------------------------------------------------------------


def test_isolated_returns_least_similar_first():
    """The compound with no close neighbor sorts to the top."""
    land = landscape(
        ids=["a", "b", "c", "d", "e", "loner"],
        fingerprints=["1111000000", "1111100000", "1111110000", "1111111000", "1111111100", "0000000011"],
        targets=[1, 2, 3, 4, 5, 6],
    )
    isolated = land.isolated(top_percent=20.0)

    assert isolated.iloc[0]["id"] == "loner"
    assert isolated["nn_similarity"].is_monotonic_increasing


def test_proximity_stats_covers_the_reference_set():
    """One stats row per percentile, counting every compound."""
    land = landscape(list("abcdef"), DISTINCT, [1, 2, 3, 4, 5, 6])
    stats = land.proximity_stats()

    assert stats.columns.tolist() == ["nn_similarity"]
    assert stats.loc["count", "nn_similarity"] == 6
