"""Tests for how SageMaker feature ingest behaves across multiprocessing start methods.

SageMaker's IngestionManagerPandas runs multiprocess ingest via a multiprocessing.Pool
whose initializer (init_worker) is a nested local function. Nested functions can't be
pickled, so multiprocess ingest only works under the 'fork' start method (Linux), where
the initializer is inherited in the child's memory. Under 'spawn' (macOS, Windows) and
'forkserver' the initializer must be pickled and ingest crashes, so
PandasToFeatures._ingest_settings() falls back to single-process ingest there.

See: https://github.com/aws/sagemaker-python-sdk/issues/5312
"""

import inspect
import multiprocessing

from workbench.core.transforms.pandas_transforms import PandasToFeatures


def test_ingest_settings():
    """_ingest_settings() enables multiprocess ingest only under the 'fork' start method."""
    max_workers, max_processes = PandasToFeatures._ingest_settings()
    if multiprocessing.get_start_method() == "fork":
        assert max_workers > 1 or max_processes > 1, "fork should use multiprocess ingest"
    else:
        assert max_workers == 1 and max_processes == 1, "spawn/forkserver should use single-process ingest"


def test_sagemaker_pool_initializer_unpicklable():
    """Tripwire for https://github.com/aws/sagemaker-python-sdk/issues/5312.

    SageMaker builds its ingest Pool with a nested-local init_worker, which is unpicklable
    and so crashes multiprocess ingest under the 'spawn'/'forkserver' start methods. We work
    around it by falling back to single-process ingest on non-fork platforms.

    This asserts the bug still exists by checking that _run_multi_process still defines
    init_worker as a local function. If it fails, AWS has likely de-nested the initializer
    and _ingest_settings() can return multiprocess values on all platforms.
    """
    from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionManagerPandas

    src = inspect.getsource(IngestionManagerPandas._run_multi_process)
    assert "def init_worker(" in src, (
        "SageMaker's _run_multi_process no longer defines a local init_worker — the spawn "
        "pickling bug may be fixed. Re-check issue #5312 and consider enabling multiprocess "
        "ingest on spawn platforms in PandasToFeatures._ingest_settings()."
    )
