"""Smoke-test the InferenceCache concurrency guarantees end-to-end.

Exercises the append-only write path, read-under-write safety, compaction,
and the graceful fallback when a dataset is poisoned by schema drift. Run
after any change to ``inference_cache.py``, ``inference_cache_utils.py``,
or the bridges ``DFStore`` append path.

Run::

    WORKBENCH_CONFIG=/path/to/config.json \\
        python scripts/admin/inference_cache_concurrency_check.py

All S3 objects created live under ``/_ic_validation/<uuid>/`` in the
WORKBENCH_BUCKET and are deleted on exit.
"""

from __future__ import annotations

import logging
import threading
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from workbench.api.df_store import DFStore
from workbench.api.inference_cache import InferenceCache

logging.basicConfig(level=logging.WARNING)

TEST_ROOT = f"/_ic_validation/{uuid.uuid4().hex[:8]}"


def banner(t: str) -> None:
    print(f"\n{'=' * 6} {t} {'=' * 6}")


def test_append_concurrent_matching():
    """4 concurrent appenders with matching schema land 4 files, 200 rows total."""
    banner("A: DFStore.append — concurrent matching-schema writers")
    store = DFStore()
    path = f"{TEST_ROOT}/A_match"
    store.delete(path)

    def w(tag: str):
        df = pd.DataFrame({"key": [f"{tag}_{i}" for i in range(50)], "val": [float(i) for i in range(50)]})
        store.append(path, df)

    with ThreadPoolExecutor(max_workers=4) as ex:
        [f.result() for f in [ex.submit(w, f"T{i}") for i in range(4)]]

    df = store.get(path)
    ok = len(df) == 200 and df["key"].str.split("_").str[0].nunique() == 4
    print(
        f"  rows={len(df)} tags={df['key'].str.split('_').str[0].nunique()} "
        f"(expect 200, 4) {'PASS' if ok else 'FAIL'}"
    )
    store.delete(path)


def test_append_reader_race():
    """Continuous reader sees no errors while a writer appends."""
    banner("B: append writer + continuous reader")
    store = DFStore()
    path = f"{TEST_ROOT}/B_race"
    store.delete(path)
    store.append(path, pd.DataFrame({"key": ["seed"], "val": [0.0]}))

    stop = threading.Event()
    errs: list[str] = []
    reads = [0]

    def reader():
        while not stop.is_set():
            try:
                if store.get(path) is not None:
                    reads[0] += 1
            except Exception as e:
                errs.append(f"R {type(e).__name__}: {e}")

    def writer():
        i = 0
        while not stop.is_set():
            df = pd.DataFrame({"key": [f"w{i}_{j}" for j in range(20)], "val": [float(j) for j in range(20)]})
            try:
                store.append(path, df)
                i += 1
            except Exception as e:
                errs.append(f"W {type(e).__name__}: {e}")

    rt, wt = threading.Thread(target=reader), threading.Thread(target=writer)
    rt.start()
    wt.start()
    time.sleep(10)
    stop.set()
    rt.join()
    wt.join()

    ok = len(errs) == 0 and reads[0] > 0
    print(f"  reads ok={reads[0]} errors={len(errs)} {'PASS' if ok else 'FAIL'}")
    for e in errs[:5]:
        print(f"    {e}")
    store.delete(path)


def _fake_endpoint(name: str, dtype_mode: str):
    ep = types.SimpleNamespace()
    ep.name = name

    def inference(chunk, **kwargs):
        out = chunk[["key"]].copy()
        if dtype_mode == "int":
            out["feat"] = [int(hash(k) % 1000) for k in out["key"]]
        else:  # "float"
            out["feat"] = [float(hash(k) % 1000) / 3.0 for k in out["key"]]
        return out

    ep.inference = inference
    ep.modified = lambda: "2026-01-01 00:00:00+00:00"
    return ep


def _fresh_cache(endpoint_name: str, cache_path: str, dtype_mode: str) -> InferenceCache:
    ep = _fake_endpoint(endpoint_name, dtype_mode)
    c = InferenceCache.__new__(InferenceCache)
    c._endpoint = ep
    c.cache_key_column = "key"
    c.output_key_column = None
    c.cache_path = cache_path
    c.manifest_path = f"{cache_path}__meta"
    c._df_store = DFStore()
    c._cache_df = None
    c._invalidation_checked = False
    c._auto_invalidate_cache = False
    c._canonical_dtypes = None
    c.chunk_size = 25
    c.log = logging.getLogger("workbench")
    return c


def test_concurrent_same_dtype_and_compact():
    """Two concurrent workers producing the same dtype → cache readable,
    types preserved, compact reduces file count without row loss."""
    banner("C: InferenceCache concurrent same-dtype + compact")
    endpoint_name = f"fake-endpoint-{uuid.uuid4().hex[:6]}"
    cache_path = f"{TEST_ROOT}/C_cache"
    store = DFStore()
    store.delete(cache_path)
    store.delete(f"{cache_path}__meta")

    # Seed with int dtype. Subsequent workers also use int → types preserved.
    seed = _fresh_cache(endpoint_name, cache_path, "int")
    seed.inference(pd.DataFrame({"key": [f"seed_{i}" for i in range(10)]}))

    def worker(tag):
        c = _fresh_cache(endpoint_name, cache_path, "int")
        c.inference(pd.DataFrame({"key": [f"{tag}_{i}" for i in range(100)]}))
        return tag, c.cache_size()

    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(worker, "A"), ex.submit(worker, "B")]
        [f.result() for f in as_completed(futs)]

    verify = _fresh_cache(endpoint_name, cache_path, "int")
    df = verify._df_store.get(cache_path)
    if df is None:
        print("  FAIL: cache unreadable")
        return
    # Types preserved: feat stays integer (whichever concrete int dtype the
    # endpoint/parquet round-trip produced).
    dtype_ok = pd.api.types.is_integer_dtype(df["feat"])
    row_ok = len(df) == 210
    print(f"  read rows={len(df)} feat_dtype={df['feat'].dtype} " f"{'PASS' if row_ok and dtype_ok else 'FAIL'}")

    before = verify.cache_size()
    after = verify.compact()
    final = verify._df_store.get(cache_path)
    dup = final["key"].duplicated().sum()
    compact_ok = after == before and dup == 0
    print(f"  compact: {before} -> {after} rows, dup_keys={dup} " f"{'PASS' if compact_ok else 'FAIL'}")

    verify._df_store.delete(cache_path)
    verify._df_store.delete(verify.manifest_path)


def test_schema_drift_graceful_fallback():
    """A cache poisoned by schema drift on disk reads as None (treated as
    empty), so callers recompute instead of crashing."""
    banner("D: schema drift → graceful fallback")
    cache_path = f"{TEST_ROOT}/D_drift"
    store = DFStore()
    store.delete(cache_path)

    # Stage two parquet files with incompatible schemas under the prefix.
    store.append(cache_path, pd.DataFrame({"key": ["a"], "feat": [1]}))  # int
    store.append(cache_path, pd.DataFrame({"key": ["b"], "feat": [1.5]}))  # float

    cache = _fresh_cache("fake", cache_path, "int")
    result = cache._read_cache_with_retry()
    # A raw DFStore.get would raise ArrowTypeError here. The cache's read
    # wrapper must detect that and return None so callers treat it as empty.
    ok = result is None
    print(
        f"  _read_cache_with_retry -> {type(result).__name__} "
        f"{'PASS (graceful fallback)' if ok else 'FAIL (should be None)'}"
    )

    store.delete(cache_path)


def main():
    try:
        test_append_concurrent_matching()
        test_append_reader_race()
        test_concurrent_same_dtype_and_compact()
        test_schema_drift_graceful_fallback()
    finally:
        try:
            DFStore().delete_recursive(TEST_ROOT)
        except Exception as e:
            print(f"cleanup warn: {e}")


if __name__ == "__main__":
    main()
