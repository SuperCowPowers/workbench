"""Unit tests for Proximity neighbor lookups."""

import pandas as pd

from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity


def count_fp_proximity(n_rows: int) -> FingerprintProximity:
    """Proximity over count fingerprints — the sparse Ruzicka backend, which clamps k."""
    df = pd.DataFrame(
        {
            "id": [f"c{i}" for i in range(n_rows)],
            "fingerprint": [",".join(str((i >> b) & 1) for b in range(8)) for i in range(n_rows)],
        }
    )
    return FingerprintProximity(df, id_column="id", fingerprint_column="fingerprint")


def test_more_neighbors_requested_than_rows():
    """Asking for more neighbors than the reference set holds returns all of them.

    The backend clamps k to the reference size, so the result assembly has to repeat
    query ids by what actually came back rather than by what was asked for.
    """
    prox = count_fp_proximity(n_rows=3)
    result = prox.neighbors("c0", n_neighbors=10)

    assert len(result) == 3
    assert set(result["neighbor_id"]) == {"c0", "c1", "c2"}


def test_more_neighbors_requested_than_rows_excluding_self():
    """Same clamp, with the self-hit filtered out."""
    prox = count_fp_proximity(n_rows=3)
    result = prox.neighbors("c0", n_neighbors=10, include_self=False)

    assert len(result) == 2
    assert "c0" not in set(result["neighbor_id"])


def test_neighbor_count_matches_request_when_set_is_large_enough():
    """The ordinary case is unaffected by the clamp."""
    prox = count_fp_proximity(n_rows=8)
    result = prox.neighbors("c0", n_neighbors=4)

    assert len(result) == 4


def test_invalid_query_smiles_keeps_other_rows_in_place():
    """A dropped unparseable SMILES leaves every other query labeled by its input position."""
    ref = pd.DataFrame(
        {
            "id": [f"r{i}" for i in range(6)],
            "smiles": ["CCO", "CCCO", "CCN", "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1"],
        }
    )
    prox = FingerprintProximity(ref, id_column="id")
    valid = prox.neighbors_from_query_df(pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]}), n_neighbors=2)
    mixed = prox.neighbors_from_query_df(pd.DataFrame({"smiles": ["CCO", "not_a_smiles", "c1ccccc1"]}), n_neighbors=2)

    assert set(mixed["query_id"]) == {0, 2}
    benzene = valid[valid["query_id"] == 1]["neighbor_id"].tolist()
    assert mixed[mixed["query_id"] == 2]["neighbor_id"].tolist() == benzene
