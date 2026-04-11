"""InferenceCache: Client-side caching wrapper around a Workbench Endpoint.

Wraps an :class:`Endpoint` and stores inference results in a shared S3-backed
:class:`DFStore` keyed on a cache-key column (SMILES by default). On each
``inference(df)`` call, rows whose cache-key value is already in the cache
are served from S3, and only the remaining rows are sent to the underlying
endpoint. Newly computed rows are written back to the cache.

Motivating use case: the ``smiles-to-3d-descriptors-v1`` feature endpoint is
slow (conformer generation + FF optimization), and the same SMILES is
frequently re-computed across calls.

Note: this is distinct from :class:`workbench.cached.CachedEndpoint`, which
caches *metadata* methods (``summary``, ``details``, ``health_check``). This
class caches *inference results*.
"""

import logging
from typing import Any, Iterable, Optional, Union

import pandas as pd

from workbench.api.df_store import DFStore
from workbench.api.endpoint import Endpoint
from workbench.utils.inference_cache_utils import (
    DEFAULT_CHUNK_SIZE,
    chunked_with_cache_writes,
)


class InferenceCache:
    """InferenceCache: Client-side caching wrapper for a Workbench Endpoint.

    Common Usage:
        ```python
        from workbench.api import Endpoint
        from workbench.api.inference_cache import InferenceCache

        endpoint = Endpoint("smiles-to-3d-descriptors-v1")
        cached = InferenceCache(endpoint, cache_key_column="smiles")

        # Drop-in replacement for endpoint.inference()
        result_df = cached.inference(eval_df)

        # Other endpoint methods still work via attribute delegation
        print(cached.name)
        cached.details()
        ```
    """

    # Rows per cache write. The endpoint is called once per chunk and the
    # cache is persisted between chunks, so this also bounds the blast radius
    # of an interrupted/failed write to one chunk worth of work. Override on
    # an instance (or via subclass) if a particular endpoint wants different
    # durability/throughput tradeoffs.
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __init__(
        self,
        endpoint: Endpoint,
        cache_key_column: str = "smiles",
    ):
        """Initialize the InferenceCache.

        Args:
            endpoint (Endpoint): The Workbench Endpoint to wrap.
            cache_key_column (str): Name of the column whose values are used
                as the cache key (default: "smiles").
        """
        self._endpoint = endpoint
        self.cache_key_column = cache_key_column
        self.cache_path = f"/workbench/inference_cache/{endpoint.name}"
        self.manifest_path = f"{self.cache_path}__meta"
        self._df_store = DFStore()
        self._cache_df: Optional[pd.DataFrame] = None  # lazy-loaded
        self._invalidation_checked = False  # per-instance, one-shot
        self.log = logging.getLogger("workbench")

    def __getattr__(self, name):
        """Delegate any unrecognized attribute access to the wrapped Endpoint."""
        # __getattr__ is only called when normal lookup fails, so this won't
        # interfere with our own attributes.
        return getattr(self._endpoint, name)

    def inference(self, eval_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Run cached inference on ``eval_df``.

        Rows whose ``cache_key_column`` value is already in the cache are
        served from S3; the rest are sent to the underlying endpoint and the
        new results are written back to the cache. The returned DataFrame
        preserves the original row order of ``eval_df``.

        Args:
            eval_df (pd.DataFrame): DataFrame to run predictions on. Must
                contain ``self.cache_key_column``.
            **kwargs (Any): Forwarded to the wrapped ``Endpoint.inference()``
                for uncached rows.

        Returns:
            pd.DataFrame: ``eval_df`` with the endpoint's added columns
            left-joined on ``cache_key_column``.
        """
        key_col = self.cache_key_column
        if key_col not in eval_df.columns:
            raise ValueError(f"eval_df is missing required cache_key_column '{key_col}'")

        cache_df = self._load_cache()

        # Split eval rows into cache hits vs rows we still need to compute
        is_cached = eval_df[key_col].isin(cache_df[key_col])
        uncached_df = eval_df[~is_cached]
        cached_hits = cache_df[cache_df[key_col].isin(eval_df[key_col])]

        hits = len(eval_df) - len(uncached_df)
        self.log.info(f"InferenceCache[{self._endpoint.name}]: {hits}/{len(eval_df)} cache hits")

        # Run the endpoint on the uncached rows. The decorator on
        # _chunked_endpoint_inference handles chunking, per-chunk cache
        # writes, and error recovery so a single failed write doesn't
        # destroy the rest of the batch.
        new_results = pd.DataFrame()
        if not uncached_df.empty:
            to_compute = uncached_df.drop_duplicates(subset=[key_col])
            new_results = self._chunked_endpoint_inference(to_compute, **kwargs)

        # Combine cached + new into a single feature table, then left-join
        # back onto eval_df to preserve row order and any extra input columns.
        # (Filter out empty frames to dodge a pandas FutureWarning about
        # dtype inference on empty entries.)
        frames = [f for f in (cached_hits, new_results) if not f.empty]
        if not frames:
            return eval_df.copy()
        feature_table = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[key_col], keep="last")
        feature_cols = [c for c in feature_table.columns if c not in eval_df.columns]
        return eval_df.merge(feature_table[[key_col] + feature_cols], on=key_col, how="left")

    # ---- cache introspection / maintenance ----
    def cache_size(self) -> int:
        """Number of rows currently in the cache."""
        return len(self._load_cache())

    def cache_info(self) -> dict:
        """Summary of the cache: path, row count, columns, manifest."""
        df = self._load_cache()
        return {
            "path": self.cache_path,
            "rows": len(df),
            "columns": list(df.columns),
            "manifest": self._load_manifest(),
        }

    def clear_cache(self) -> None:
        """Delete the cache (and manifest) from S3 and reset in-memory state."""
        if self._df_store.check(self.cache_path):
            self._df_store.delete(self.cache_path)
        if self._df_store.check(self.manifest_path):
            self._df_store.delete(self.manifest_path)
        self._cache_df = pd.DataFrame(columns=[self.cache_key_column])

    def delete_entries(self, keys: Union[Any, Iterable[Any]]) -> int:
        """Remove one or more entries from the cache by cache-key value(s).

        Use this to drop bad results that should be recomputed on the next
        ``inference()`` call.

        Args:
            keys (Union[Any, Iterable[Any]]): A single cache-key value, or an
                iterable of them.

        Returns:
            int: Number of rows removed from the cache.
        """
        if isinstance(keys, (str, bytes)) or not hasattr(keys, "__iter__"):
            keys = [keys]
        keys = list(keys)

        cache_df = self._load_cache()
        if cache_df.empty:
            return 0

        keep_mask = ~cache_df[self.cache_key_column].isin(keys)
        removed = int((~keep_mask).sum())
        if removed == 0:
            return 0

        new_cache = cache_df[keep_mask].reset_index(drop=True)
        if new_cache.empty:
            # Nothing left — delete the cache file entirely but keep the manifest
            if self._df_store.check(self.cache_path):
                self._df_store.delete(self.cache_path)
        else:
            self._df_store.upsert(self.cache_path, new_cache)
        self._cache_df = new_cache
        self.log.info(f"InferenceCache[{self._endpoint.name}]: removed {removed} entries")
        return removed

    # ---- internals ----

    def _load_cache(self) -> pd.DataFrame:
        """Lazily load the cache DataFrame from DFStore.

        If the cache doesn't yet exist, returns an empty DataFrame that
        still has ``cache_key_column`` defined, so callers can always do
        ``df[cache_key_column]`` without special-casing the empty case.

        On first call, also checks whether the endpoint has been modified
        since the cache was written and auto-invalidates if so.
        """
        if self._cache_df is None:
            if not self._invalidation_checked:
                self._check_endpoint_changed()
                self._invalidation_checked = True

            df = self._df_store.get(self.cache_path)
            if df is None:
                df = pd.DataFrame(columns=[self.cache_key_column])
            self._cache_df = df
        return self._cache_df

    @chunked_with_cache_writes
    def _chunked_endpoint_inference(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run the wrapped endpoint on one chunk of rows.

        The :func:`chunked_with_cache_writes` decorator handles chunking,
        per-chunk persistence via :meth:`_update_cache`, and error recovery.
        """
        return self._endpoint.inference(chunk, **kwargs)

    def _update_cache(self, new_results: pd.DataFrame) -> None:
        """Merge new results into the in-memory cache and persist to DFStore.

        Reads the current in-memory cache (``self._cache_df``), appends the
        new rows, dedups on ``cache_key_column``, writes the combined frame
        back to S3, and refreshes the manifest. Called once per chunk by the
        decorator on :meth:`_chunked_endpoint_inference`.
        """
        if new_results.empty:
            return
        old_cache = self._cache_df if self._cache_df is not None else pd.DataFrame(columns=[self.cache_key_column])
        # Filter empty frames before concat to dodge the pandas FutureWarning
        # about dtype inference on empty entries.
        frames = [f for f in (old_cache, new_results) if not f.empty]
        combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset=[self.cache_key_column], keep="last")
        self._df_store.upsert(self.cache_path, combined)
        self._save_manifest()
        self._cache_df = combined

    # ---- endpoint-change detection ----

    def _current_endpoint_modified(self) -> Optional[str]:
        """Read the endpoint's current 'modified' timestamp.

        Stringified so the comparison is robust to tz-aware/naive datetime
        round-tripping through parquet.
        """
        try:
            modified = self._endpoint.modified()
        except Exception as e:
            self.log.warning(
                f"InferenceCache[{self._endpoint.name}]: could not read "
                f"endpoint modified time for change detection: {e}"
            )
            return None
        return str(modified) if modified is not None else None

    def _load_manifest(self) -> Optional[dict]:
        """Load the sidecar manifest (or None if it doesn't exist)."""
        df = self._df_store.get(self.manifest_path)
        if df is None or df.empty:
            return None
        return df.iloc[0].to_dict()

    def _save_manifest(self) -> None:
        """Write the sidecar manifest capturing the endpoint's current state."""
        manifest_df = pd.DataFrame(
            [
                {
                    "endpoint_name": self._endpoint.name,
                    "endpoint_modified": self._current_endpoint_modified(),
                    "cache_key_column": self.cache_key_column,
                }
            ]
        )
        self._df_store.upsert(self.manifest_path, manifest_df)

    def _check_endpoint_changed(self) -> None:
        """Compare the stored manifest against the endpoint's current modified time.

        - If no manifest exists, seed one (first run after a clean slate).
        - If the stored and current modified times differ, warn and clear
          the cache so the next call recomputes from scratch.
        """
        manifest = self._load_manifest()
        current = self._current_endpoint_modified()

        if manifest is None:
            # No manifest yet — seed one if there's already a cache, so the
            # next check has something to compare against. (If there's no
            # cache either, the manifest will be written on first update.)
            if self._df_store.check(self.cache_path) and current is not None:
                self._save_manifest()
            return

        stored = manifest.get("endpoint_modified")
        if stored is None or current is None or stored == current:
            return

        self.log.warning(
            f"InferenceCache[{self._endpoint.name}]: endpoint was modified "
            f"since cache was written (stored={stored}, current={current}). "
            f"Auto-invalidating cache."
        )
        self.clear_cache()


if __name__ == "__main__":
    """Exercise the InferenceCache class against the 3D descriptors endpoint."""
    import time

    from workbench.api import Endpoint, FeatureSet

    endpoint = Endpoint("smiles-to-3d-descriptors-v1")
    cached = InferenceCache(endpoint, cache_key_column="smiles")

    # Start clean so the timing comparison is meaningful
    cached.clear_cache()

    df = FeatureSet("feature_endpoint_fs").pull_dataframe()[:50]
    print(f"Running cached inference on {len(df)} rows ({df['smiles'].nunique()} unique SMILES)")

    # First call: cache is empty, full endpoint run
    t0 = time.time()
    r1 = cached.inference(df)
    t1 = time.time()
    print(f"first call:  {t1 - t0:5.1f}s, cache now has {cached.cache_size()} rows")

    # Second call: should be all hits, near-instant
    t0 = time.time()
    r2 = cached.inference(df)
    t1 = time.time()
    print(f"second call: {t1 - t0:5.1f}s (should be ~all cache hits)")

    # Partial hit: first 50 are cached, next 25 are new
    df2 = FeatureSet("feature_endpoint_fs").pull_dataframe()[:75]
    t0 = time.time()
    r3 = cached.inference(df2)
    t1 = time.time()
    print(
        f"partial:     {t1 - t0:5.1f}s, cache now has {cached.cache_size()} rows "
        f"(expected ~{df2['smiles'].nunique()})"
    )

    print("\nCache info:")
    print(cached.cache_info())

    print("\nResult shapes:")
    print(f"  r1: {r1.shape}")
    print(f"  r2: {r2.shape}")
    print(f"  r3: {r3.shape}")

    # Exercise delete_entries: drop the first 5 SMILES, they'll recompute next time
    victims = df["smiles"].iloc[:5].tolist()
    removed = cached.delete_entries(victims)
    print(f"\ndelete_entries removed {removed} rows; cache now {cached.cache_size()}")
