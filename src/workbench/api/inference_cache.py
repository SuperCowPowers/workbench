"""InferenceCache: Client-side caching wrapper around a Workbench Endpoint.

Wraps an ``Endpoint`` and stores inference results in a shared S3-backed
``DFStore`` keyed on a cache-key column (SMILES by default). On each
``inference(df)`` call, rows whose cache-key value is already in the cache
are served from S3, and only the remaining rows are sent to the underlying
endpoint. Newly computed rows are written back to the cache.

Motivating use case: the ``smiles-to-3d-fast-v1`` feature endpoint is
slow (conformer generation + FF optimization), and the same SMILES is
frequently re-computed across calls.

Note: this is distinct from ``workbench.cached.CachedEndpoint``, which
caches *metadata* methods (``summary``, ``details``, ``health_check``). This
class caches *inference results*.
"""

import logging
import time
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

        endpoint = Endpoint("smiles-to-3d-fast-v1")
        cached_endpoint = InferenceCache(endpoint, cache_key_column="smiles")

        # Drop-in replacement for endpoint.inference()
        result_df = cached_endpoint.inference(eval_df)

        # Other endpoint methods still work via attribute delegation
        print(cached_endpoint.name)
        cached_endpoint.details()
        ```
    """

    # Rows per cache write. The endpoint is called once per chunk and the
    # cache is persisted between chunks, so this also bounds the blast radius
    # of an interrupted/failed write to one chunk worth of work.
    #
    # The actual chunk_size on each instance is set in __init__: either the
    # explicit ``chunk_size`` constructor kwarg, or — for async endpoints with
    # max_instances in their workbench_meta — derived from fleet capacity
    # (max_instances × batch_size × 2) so each chunk holds an integer number
    # of full fleet-waves. This avoids the "10 batches / 8 instances → tail"
    # utilization loss. Falls back to this class attribute (DEFAULT_CHUNK_SIZE)
    # for sync endpoints or legacy endpoints without max_instances in meta.
    chunk_size: int = DEFAULT_CHUNK_SIZE

    # Number of fleet-waves per chunk when auto-deriving chunk_size. k=2 is a
    # compromise: small enough to bound crash-recovery loss (1 chunk = 2 waves
    # of work), large enough that partial-last-wave is a diminishing fraction.
    _CHUNK_WAVES = 2

    def __init__(
        self,
        endpoint: Endpoint,
        cache_key_column: str = "smiles",
        output_key_column: Optional[str] = None,
        auto_invalidate_cache: bool = False,
        chunk_size: Optional[int] = None,
    ):
        """Initialize the InferenceCache.

        Args:
            endpoint (Endpoint): The Workbench Endpoint to wrap.
            cache_key_column (str): Name of the column whose values are used
                as the cache key (default: "smiles").
            output_key_column (Optional[str]): Name of the column in the
                endpoint's *output* that contains the original input key
                values. Some endpoints normalize/canonicalize the key column
                (e.g. canonical SMILES) and place the original value in a
                separate column (e.g. "orig_smiles"). When set, the cache
                uses this column's values as the key so future lookups with
                the original input values still hit. When None (default),
                the cache key column in the output is assumed to match the
                input unchanged.
            auto_invalidate_cache (bool): When True, automatically clear the
                cache if the endpoint has been modified since the cache was
                last written. When False (default), the existing cache is
                kept regardless of endpoint changes — the manifest is
                reseeded on first load so subsequent calls have a consistent
                baseline.
            chunk_size (Optional[int]): Rows per cache write. If ``None``
                (default), derived from the endpoint's ``max_instances`` and
                ``inference_batch_size`` to produce full fleet-waves — see
                :meth:`_derive_chunk_size`. Falls back to
                ``DEFAULT_CHUNK_SIZE`` when fleet info isn't available.
        """
        self._endpoint = endpoint
        self.cache_key_column = cache_key_column
        self.output_key_column = output_key_column
        self.cache_path = f"/workbench/inference_cache/{endpoint.name}"
        self.manifest_path = f"{self.cache_path}__meta"
        self._df_store = DFStore()
        self._cache_df: Optional[pd.DataFrame] = None  # lazy-loaded
        self._invalidation_checked = False  # per-instance, one-shot
        self._auto_invalidate_cache = auto_invalidate_cache
        # Canonical dtype map for the cache, captured on first non-empty load
        # and used to coerce subsequent appended chunks so concurrent writers
        # never produce a schema-incompatible dataset.
        self._canonical_dtypes: Optional[pd.Series] = None
        self.log = logging.getLogger("workbench")

        # Resolve chunk_size: explicit override wins; else try fleet-derivation;
        # else fall through to the class-level DEFAULT_CHUNK_SIZE.
        if chunk_size is not None:
            self.chunk_size = int(chunk_size)
        else:
            derived = self._derive_chunk_size()
            if derived is not None:
                self.chunk_size = derived

    def _derive_chunk_size(self) -> Optional[int]:
        """Derive chunk_size from the wrapped endpoint's fleet capacity.

        Returns ``max_instances × batch_size × _CHUNK_WAVES`` so each chunk
        holds an integer number of full fleet-waves — preventing the
        "10 batches / 8 instances → 2-batch tail" utilization loss on async
        endpoints. Returns ``None`` (→ caller should use DEFAULT_CHUNK_SIZE)
        when the endpoint's meta doesn't have ``max_instances``, which is
        the case for sync endpoints and legacy async deploys.
        """
        try:
            meta = self._endpoint.workbench_meta() or {}
        except Exception:
            return None
        max_instances = meta.get("max_instances")
        if max_instances is None:
            return None
        # Mirror AsyncEndpoint's own resolution: explicit meta override wins,
        # otherwise the core default.
        from workbench.core.artifacts.async_endpoint_core import _DEFAULT_BATCH_SIZE

        batch_size = int(meta.get("inference_batch_size", _DEFAULT_BATCH_SIZE))
        derived = int(max_instances) * batch_size * self._CHUNK_WAVES
        self.log.info(
            f"InferenceCache[{self._endpoint.name}]: chunk_size={derived} "
            f"(max_instances={max_instances} × batch_size={batch_size} × "
            f"{self._CHUNK_WAVES} waves — full fleet utilization per chunk)"
        )
        return derived

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
                if self._auto_invalidate_cache:
                    self._check_endpoint_changed()
                else:
                    # Skip the auto-invalidation check and reseed the manifest
                    # so the stored modified time matches the current endpoint.
                    self.log.info(
                        f"InferenceCache[{self._endpoint.name}]: auto_invalidate_cache=False, "
                        f"reseeding manifest and keeping existing cache"
                    )
                    if self._df_store.check(self.cache_path):
                        self._save_manifest()
                self._invalidation_checked = True

            df = self._read_cache_with_retry()
            if df is None:
                df = pd.DataFrame(columns=[self.cache_key_column])
            if not df.empty:
                self._canonical_dtypes = df.dtypes
            self._cache_df = df
        return self._cache_df

    def _read_cache_with_retry(self, attempts: int = 3, backoff: float = 0.5) -> Optional[pd.DataFrame]:
        """Read the cache, tolerating transient and schema-mismatch failures.

        Handles two distinct error classes:

        - **Transient** (e.g. ``NoSuchKey`` from a concurrent overwrite or
          compaction) — bounded retries with short backoff. The cache may
          recover within a couple of seconds.
        - **Schema-mismatch** (PyArrow ``ArrowTypeError`` / ``ArrowInvalid``
          from incompatible parquet files under the dataset prefix) — not
          transient; don't retry. Log and return ``None`` so the caller
          treats the cache as empty and the affected rows recompute on the
          next inference call. Run :meth:`clear_cache` or inspect the files
          manually to resolve; :meth:`compact` also reads through the same
          path so it will not self-heal this case.
        """
        try:
            import pyarrow as pa  # deferred import; pa is transitively installed

            schema_errs: tuple = (pa.lib.ArrowTypeError, pa.lib.ArrowInvalid)
        except Exception:
            schema_errs = ()

        last_err: Optional[Exception] = None
        for i in range(attempts):
            try:
                return self._df_store.get(self.cache_path)
            except schema_errs as e:
                self.log.error(
                    f"InferenceCache[{self._endpoint.name}]: cache is schema-"
                    f"incompatible ({type(e).__name__}: {e}). Treating as empty; "
                    f"rows will recompute. Run clear_cache() to reset."
                )
                return None
            except Exception as e:
                last_err = e
                self.log.warning(
                    f"InferenceCache[{self._endpoint.name}]: cache read failed "
                    f"(attempt {i + 1}/{attempts}): {type(e).__name__}: {e}"
                )
                time.sleep(backoff * (i + 1))
        self.log.error(
            f"InferenceCache[{self._endpoint.name}]: cache read failed after "
            f"{attempts} attempts, treating as empty: {last_err}"
        )
        return None

    @chunked_with_cache_writes
    def _chunked_endpoint_inference(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run the wrapped endpoint on one chunk of rows.

        The :func:`chunked_with_cache_writes` decorator handles chunking,
        per-chunk persistence via :meth:`_update_cache`, and error recovery.
        """
        return self._endpoint.inference(chunk, **kwargs)

    def _update_cache(self, new_results: pd.DataFrame) -> None:
        """Persist ``new_results`` as a new file under the cache prefix.

        Uses :meth:`DFStore.append` so concurrent writers each land a distinct
        parquet file under the dataset prefix — eliminating the delete-then-
        write race of the old overwrite-based approach. The in-memory view
        (``self._cache_df``) is updated by concat+dedup so subsequent chunks
        in this process skip rows this worker just computed, but S3 only
        receives the new slice. Dtypes are coerced to the canonical schema
        so concurrently-appended files remain Arrow-mergeable on read.
        """
        if new_results.empty:
            return

        to_write = self._coerce_to_canonical(new_results)

        self._df_store.append(self.cache_path, to_write)
        self._save_manifest()

        # Update the local view so this worker's later chunks see these rows.
        old_cache = self._cache_df if self._cache_df is not None else pd.DataFrame(columns=[self.cache_key_column])
        frames = [f for f in (old_cache, to_write) if not f.empty]
        self._cache_df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=[self.cache_key_column], keep="last"
        )

    def _coerce_to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast ``df`` columns to match the canonical schema when one exists.

        Types are preserved as the endpoint produces them. The first non-empty
        load (or first append on a fresh cache) seeds ``self._canonical_dtypes``;
        later writes cast to match so a single worker that observes slightly
        different dtype inference per chunk stays consistent with the cache
        already on disk. No widening is performed — if an incoming column
        can't be cast losslessly to the canonical dtype, a warning is logged
        and the write proceeds with the original dtype. If that happens
        across workers the dataset can become schema-incompatible on read;
        :meth:`_read_cache_with_retry` detects that and falls back to an
        empty-cache result so the affected rows simply recompute.
        """
        if self._canonical_dtypes is None:
            self._canonical_dtypes = df.dtypes
            return df

        out = df.copy()
        for col, dtype in self._canonical_dtypes.items():
            if col not in out.columns:
                continue
            if out[col].dtype == dtype:
                continue
            try:
                out[col] = out[col].astype(dtype)
            except Exception as e:
                self.log.warning(
                    f"InferenceCache[{self._endpoint.name}]: could not coerce "
                    f"column '{col}' from {out[col].dtype} to {dtype}: {e}"
                )
        return out

    def compact(self) -> int:
        """Merge all per-chunk append files into a single deduped file.

        Append-only writes accumulate one file per ``_update_cache`` call. Over
        time this inflates S3 object count, list costs, and read latency.
        ``compact()`` reads the whole cache (as one dataset), dedups on
        ``cache_key_column``, and rewrites it via ``upsert`` (which uses
        ``mode="overwrite"``). Expected cadence: weekly / monthly as a
        maintenance op, not on the hot inference path. Do not run during
        active inference traffic — the rewrite races with concurrent
        appenders the same way any overwrite does, and can lose recent rows
        (they'll just be recomputed on the next call).

        Returns:
            int: Row count after compaction.
        """
        df = self._read_cache_with_retry()
        if df is None or df.empty:
            self.log.info(f"InferenceCache[{self._endpoint.name}]: nothing to compact")
            return 0

        before = len(df)
        df = df.drop_duplicates(subset=[self.cache_key_column], keep="last").reset_index(drop=True)
        after = len(df)

        self._df_store.upsert(self.cache_path, df)
        self._save_manifest()
        self._cache_df = df
        self._canonical_dtypes = df.dtypes

        self.log.info(f"InferenceCache[{self._endpoint.name}]: compacted {before} -> {after} rows")
        return after

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
    from workbench.api import Endpoint, FeatureSet

    endpoint = Endpoint("smiles-to-3d-fast-v1")
    cached_endpoint = InferenceCache(endpoint, cache_key_column="smiles")

    # Start clean so the timing comparison is meaningful
    cached_endpoint.clear_cache()

    df = FeatureSet("feature_endpoint_fs").pull_dataframe()[:50]
    print(f"Running cached inference on {len(df)} rows ({df['smiles'].nunique()} unique SMILES)")

    # First call: cache is empty, full endpoint run
    t0 = time.time()
    r1 = cached_endpoint.inference(df)
    t1 = time.time()
    print(f"first call:  {t1 - t0:5.1f}s, cache now has {cached_endpoint.cache_size()} rows")

    # Second call: should be all hits, near-instant
    t0 = time.time()
    r2 = cached_endpoint.inference(df)
    t1 = time.time()
    print(f"second call: {t1 - t0:5.1f}s (should be ~all cache hits)")

    # Partial hit: first 50 are cached, next 25 are new
    df2 = FeatureSet("feature_endpoint_fs").pull_dataframe()[:75]
    t0 = time.time()
    r3 = cached_endpoint.inference(df2)
    t1 = time.time()
    print(
        f"partial:     {t1 - t0:5.1f}s, cache now has {cached_endpoint.cache_size()} rows "
        f"(expected ~{df2['smiles'].nunique()})"
    )

    print("\nCache info:")
    print(cached_endpoint.cache_info())

    print("\nResult shapes:")
    print(f"  r1: {r1.shape}")
    print(f"  r2: {r2.shape}")
    print(f"  r3: {r3.shape}")

    # Exercise delete_entries: drop the first 5 SMILES, they'll recompute next time
    victims = df["smiles"].iloc[:5].tolist()
    removed = cached_endpoint.delete_entries(victims)
    print(f"\ndelete_entries removed {removed} rows; cache now {cached_endpoint.cache_size()}")
