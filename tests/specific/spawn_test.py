"""Test whether SageMaker V3 ingest_dataframe works with max_workers/max_processes > 1 on macOS.

Background: macOS Tahoe 26+ broke forked processes that create boto3 sessions. The SageMaker V2
FeatureGroup.ingest() used multiprocessing with fork, which would hang. We had a mac_spawn_hack
workaround that forced spawn mode. When we ported to SageMaker V3, we removed the hack and pinned
max_workers=1, max_processes=1 to sidestep the issue entirely.

This test checks whether V3's ingest_dataframe actually works with multiprocessing (values > 1)
without hanging or crashing. If it passes, we can bump the values back up for performance.

Current status (2026-03-31): FAILS on macOS — V3's _run_multi_process defines a local function
(init_worker) that can't be pickled under spawn mode, so it crashes with AttributeError.
Python 3.13 on macOS defaults to spawn, so this is broken out of the box.
PandasToFeatures._ingest_settings() detects macOS and falls back to (1, 1).

See: https://github.com/aws/sagemaker-python-sdk/issues/5312
"""

import platform
import signal
import pytest
from sagemaker.mlops.feature_store import ingest_dataframe
from workbench.api import FeatureSet
from workbench.core.transforms.pandas_transforms import PandasToFeatures
from workbench.utils.synthetic_data_generator import SyntheticDataGenerator


def test_ingest_settings():
    """Verify that _ingest_settings returns (1, 1) on macOS and higher values elsewhere."""
    max_workers, max_processes = PandasToFeatures._ingest_settings()
    if platform.system() == "Darwin":
        assert max_workers == 1 and max_processes == 1, "macOS should use single-process ingest"
    else:
        assert max_workers > 1 or max_processes > 1, "Linux should use multiprocess ingest"

FEATURE_SET_NAME = "spawn_test_temp"
INGEST_TIMEOUT = 120  # seconds — if it hangs longer than this, the fork issue is still present


class IngestTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise IngestTimeout(f"Ingest hung for >{INGEST_TIMEOUT}s — macOS fork/spawn issue likely still present in V3")


def _ensure_feature_set():
    """Create the temp feature set if it doesn't already exist (takes ~10 min)."""
    fs = FeatureSet(FEATURE_SET_NAME)
    if fs.exists():
        return fs.pull_dataframe()[:100]  # Return existing data for re-ingest test

    # Generate a small DataFrame
    test_data = SyntheticDataGenerator()
    df = test_data.ml_data(n_samples=100, n_features=4)
    df.insert(0, "id", range(len(df)))

    # Create via PandasToFeatures (workers=1 — known safe)
    to_features = PandasToFeatures(FEATURE_SET_NAME)
    to_features.set_input(df, id_column="id")
    to_features.set_output_tags(["test", "temp", "spawn_test"])
    to_features.transform()
    return df


@pytest.mark.long
def test_multiprocess_ingest():
    """Test that ingest_dataframe works with max_workers=4, max_processes=4 without hanging."""

    # Only meaningful on macOS, but run everywhere for correctness
    if platform.system() != "Darwin":
        pytest.skip("Fork/spawn issue is macOS-specific — skipping on this platform")

    df = _ensure_feature_set()

    # Test the multiprocess ingest with a timeout so we don't hang forever
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(INGEST_TIMEOUT)
    try:
        ingest_dataframe(
            feature_group_name=FEATURE_SET_NAME,
            data_frame=df,
            max_workers=4,
            max_processes=4,
            wait=True,
        )
        print("SUCCESS: multiprocess ingest completed without hanging")
    except IngestTimeout:
        pytest.fail(f"Ingest hung for >{INGEST_TIMEOUT}s — macOS fork/spawn issue still present in SageMaker V3")
    except AttributeError as e:
        if "init_worker" in str(e):
            pytest.fail(
                f"SageMaker V3 multiprocess ingest broken under spawn mode: {e}\n"
                "The local init_worker function in _run_multi_process can't be pickled.\n"
                "Keep max_workers=1, max_processes=1 until AWS fixes this."
            )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == "__main__":
    test_multiprocess_ingest()
