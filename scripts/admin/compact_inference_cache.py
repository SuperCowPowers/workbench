"""Compact the InferenceCache dataset for the 3D feature endpoints.

Append-only writes accumulate one parquet file per chunk. Over time this
inflates S3 object count and read latency. This script calls
:meth:`InferenceCache.compact` to merge all files into a single deduped
parquet. Expected cadence: weekly / monthly. Do NOT run during active
inference traffic — the rewrite races with concurrent appenders and can
drop in-flight rows (they'll recompute on the next inference call).

Usage::

    WORKBENCH_CONFIG=/path/to/config.json \\
        python scripts/admin/compact_inference_cache.py

    # Subset:
    python scripts/admin/compact_inference_cache.py --endpoint smiles-to-3d-fast-v1
"""

import argparse
import logging

from workbench.api import Endpoint
from workbench.api.inference_cache import InferenceCache

log = logging.getLogger("workbench")
logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_ENDPOINTS = [
    "smiles-to-3d-fast-v1",
    "smiles-to-3d-full-v1",
]


def compact_one(endpoint_name: str) -> None:
    print(f"\n=== {endpoint_name} ===")
    try:
        endpoint = Endpoint(endpoint_name)
    except Exception as e:
        print(f"  skip: could not load endpoint ({type(e).__name__}: {e})")
        return

    cache = InferenceCache(endpoint, cache_key_column="smiles")

    info_before = cache.cache_info()
    rows_before = info_before["rows"]
    cols = info_before["columns"]
    print(f"  before: {rows_before} rows, {len(cols)} columns")

    if rows_before == 0:
        print("  nothing to compact")
        return

    rows_after = cache.compact()
    print(f"  after:  {rows_after} rows  (removed {rows_before - rows_after} duplicates)")


def main():
    ap = argparse.ArgumentParser(description="Compact InferenceCache datasets.")
    ap.add_argument(
        "--endpoint",
        action="append",
        help="Endpoint name to compact (repeatable). Defaults to both 3D endpoints.",
    )
    args = ap.parse_args()

    endpoints = args.endpoint or DEFAULT_ENDPOINTS
    for name in endpoints:
        compact_one(name)


if __name__ == "__main__":
    main()
