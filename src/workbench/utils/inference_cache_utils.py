"""Helpers for the InferenceCache class.

The motivating use case is bounding the blast radius of mid-batch failures
on long-running endpoint inference jobs. SSO tokens expire, network links
hiccup, and a single failed S3 write can otherwise destroy hours of compute.

The :func:`chunked_with_cache_writes` decorator wraps a bound method that
performs inference on a single snapshot's worth of rows and turns it into
one that:

1. Slices its input DataFrame into snapshots of ``self.snapshot`` rows.
2. Calls the wrapped method on each snapshot.
3. After each snapshot, calls ``self._update_cache(results)`` to persist.
4. Catches exceptions from both inference and cache writes — a single bad
   snapshot does not kill the rest of the batch.
5. Concatenates surviving snapshots and returns the merged DataFrame.
"""

import functools
import logging
from typing import Callable

import pandas as pd

log = logging.getLogger("workbench")


def chunked_with_cache_writes(method: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Decorator: chunk a DataFrame argument and persist after each snapshot.

    The decorated method must have the signature
    ``method(self, chunk_df: pd.DataFrame, **kwargs) -> pd.DataFrame``
    and the host class must provide:

    - ``self.snapshot`` (int): rows per snapshot/chunk
    - ``self._update_cache(new_results: pd.DataFrame) -> None``: persist hook
    - ``self._endpoint.name`` (str): used in log messages
    - ``self.log``: a workbench logger

    Returns the concatenation of every successful snapshot's results. May be
    shorter than the input if any snapshots failed (failures are logged at
    ERROR level).
    """

    @functools.wraps(method)
    def wrapper(self, to_compute: pd.DataFrame, **kwargs) -> pd.DataFrame:
        total = len(to_compute)
        if total == 0:
            return to_compute.iloc[:0]

        snapshot = self.snapshot
        label = f"InferenceCache[{self._endpoint.name}]"
        n_chunks = (total + snapshot - 1) // snapshot
        self.log.info(f"{label}: chunking {total} rows into {n_chunks} snapshots of {snapshot}")

        results = []
        for i in range(n_chunks):
            chunk = to_compute.iloc[i * snapshot : (i + 1) * snapshot]
            sent = len(chunk)
            self.log.info(f"{label}: chunk {i + 1}/{n_chunks} ({sent} rows)")

            # Step 1: run inference on the chunk
            try:
                chunk_results = method(self, chunk, **kwargs)
            except Exception as e:
                self.log.error(f"{label}: chunk {i + 1}/{n_chunks} inference failed: {e}")
                continue

            got = len(chunk_results)
            if got < sent:
                # The endpoint should always return one row per input row.
                # Anything less means something went wrong — log loudly but
                # don't raise so the rest of the batch keeps going.
                self.log.error(
                    f"{label}: chunk {i + 1}/{n_chunks} sent {sent} rows but got "
                    f"{got} back ({sent - got} missing). Missing keys will be "
                    f"re-requested on the next call."
                )

            # If the endpoint normalizes the key column (e.g. canonical
            # SMILES) and places the original values in output_key_column,
            # swap them so the cache is keyed on the original input values.
            key_col = getattr(self, "cache_key_column", None)
            output_key_col = getattr(self, "output_key_column", None)
            if output_key_col and output_key_col in chunk_results.columns:
                chunk_results = chunk_results.copy()
                chunk_results[key_col] = chunk_results[output_key_col]

            # Step 2: persist only the key + endpoint feature columns to the
            # cache (drop the caller's input columns so they don't leak to
            # other callers who hit the cache with different input schemas).
            if not chunk_results.empty:
                try:
                    input_cols = set(chunk.columns)
                    feature_cols = [c for c in chunk_results.columns if c not in input_cols]
                    cache_cols = [key_col] + feature_cols if key_col else list(chunk_results.columns)
                    self._update_cache(chunk_results[cache_cols])
                except Exception as e:
                    self.log.error(
                        f"{label}: chunk {i + 1}/{n_chunks} cache write failed: {e}. "
                        f"Results returned to caller but not cached — will be "
                        f"re-requested on the next call."
                    )

            results.append(chunk_results)

        # Filter empty frames before concat to dodge the pandas FutureWarning
        # about dtype inference on empty entries.
        frames = [f for f in results if not f.empty]
        if not frames:
            return to_compute.iloc[:0]
        return pd.concat(frames, ignore_index=True)

    return wrapper
