"""Helpers for the InferenceCache class.

The motivating use case is bounding the blast radius of mid-batch failures
on long-running endpoint inference jobs. SSO tokens expire, network links
hiccup, and a single failed S3 write can otherwise destroy hours of compute.

The :func:`chunked_with_cache_writes` decorator wraps a bound method that
performs inference on a single chunk and turns it into one that:

1. Slices its input DataFrame into chunks of ``self.chunk_size`` rows.
2. Calls the wrapped method on each chunk.
3. After each chunk, calls ``self._update_cache(chunk_results)`` to persist.
4. Catches exceptions from both inference and cache writes — a single bad
   chunk does not kill the rest of the batch.
5. Concatenates surviving chunks and returns the merged DataFrame.
"""

import functools
import logging
from typing import Callable

import pandas as pd

log = logging.getLogger("workbench")

DEFAULT_CHUNK_SIZE = 500


def chunked_with_cache_writes(method: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """Decorator: chunk a DataFrame argument and persist after each chunk.

    The decorated method must have the signature
    ``method(self, chunk_df: pd.DataFrame, **kwargs) -> pd.DataFrame``
    and the host class must provide:

    - ``self.chunk_size`` (int): rows per chunk
    - ``self._update_cache(new_results: pd.DataFrame) -> None``: persist hook
    - ``self._endpoint.name`` (str): used in log messages
    - ``self.log``: a workbench logger

    Returns the concatenation of every successful chunk's results. May be
    shorter than the input if any chunks failed (failures are logged at
    ERROR level).
    """

    @functools.wraps(method)
    def wrapper(self, to_compute: pd.DataFrame, **kwargs) -> pd.DataFrame:
        total = len(to_compute)
        if total == 0:
            return to_compute.iloc[:0]

        chunk_size = getattr(self, "chunk_size", DEFAULT_CHUNK_SIZE)
        label = f"InferenceCache[{self._endpoint.name}]"
        n_chunks = (total + chunk_size - 1) // chunk_size
        self.log.info(
            f"{label}: chunking {total} rows into {n_chunks} chunks of {chunk_size}"
        )

        results = []
        for i in range(n_chunks):
            chunk = to_compute.iloc[i * chunk_size : (i + 1) * chunk_size]
            sent = len(chunk)
            self.log.info(f"{label}: chunk {i + 1}/{n_chunks} ({sent} rows)")

            # Step 1: run inference on the chunk
            try:
                chunk_results = method(self, chunk, **kwargs)
            except Exception as e:
                self.log.error(
                    f"{label}: chunk {i + 1}/{n_chunks} inference failed: {e}"
                )
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

            # Step 2: persist this chunk's results to the cache
            if not chunk_results.empty:
                try:
                    self._update_cache(chunk_results)
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
