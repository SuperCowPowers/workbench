"""Unit test for what the InferenceCache is allowed to remember.

Local + fast (no AWS). A failed async chunk comes back as its own input rows with
the endpoint's columns NaN-filled, so the row count never reveals the failure.
Caching such a row would answer every later lookup for that key from the cache, so
the row would never recompute and a retryable failure would become a permanent one.
"""

import logging

import pandas as pd

from workbench.utils.inference_cache_utils import chunked_with_cache_writes


class FakeCache:
    """Minimal host for the decorator, recording whatever it is asked to cache."""

    snapshot = 100
    cache_key_column = "smiles"
    output_key_column = None

    def __init__(self, scored_ids):
        self._scored = set(scored_ids)
        self._endpoint = type("E", (), {"name": "fake-endpoint"})()
        self.log = logging.getLogger("workbench")
        self.cached = pd.DataFrame()

    def _update_cache(self, new_results):
        self.cached = pd.concat([self.cached, new_results], ignore_index=True)

    @chunked_with_cache_writes
    def run(self, chunk):
        """Score the rows in `_scored` and pass the rest through NaN, as async does."""
        out = chunk.copy()
        out["prediction"] = [1.23 if i in self._scored else float("nan") for i in chunk["id"]]
        return out


def frame(n: int) -> pd.DataFrame:
    return pd.DataFrame({"id": range(n), "smiles": [f"C{i}" for i in range(n)]})


def test_unscored_rows_are_not_cached():
    """Half the chunk comes back empty: only the scored half is remembered."""
    cache = FakeCache(scored_ids=range(5))

    raised = False
    try:
        cache.run(frame(10))
    except RuntimeError:
        raised = True

    assert raised, "unscored rows must raise so the caller retries them"
    assert sorted(cache.cached["smiles"]) == [f"C{i}" for i in range(5)]
    assert cache.cached["prediction"].notna().all()


def test_fully_scored_chunk_caches_everything():
    """The happy path is untouched."""
    cache = FakeCache(scored_ids=range(10))

    out = cache.run(frame(10))

    assert len(out) == 10
    assert len(cache.cached) == 10
