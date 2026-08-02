"""Tests for the training-job utilization reader in model_utils (live Workbench/AWS).

Run against the sandbox:
    WORKBENCH_CONFIG=~/.workbench/scp_sandbox.json .venv/bin/python -m pytest tests/utils/test_training_utilization.py

Marked `medium` — each test describes a training job and queries CloudWatch.

CloudWatch keeps 1-minute datapoints for 15 days, so these skip rather than fail once a
fixture's training job ages out. Regenerating the fixtures brings them back:
test_artifacts/create_aqsol_artifacts.py and ml_pipelines/Testing/AqSol/aqsol_chemprop_hpo.py.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

# Workbench Imports
from workbench.api import Model
from workbench.utils.training_job_utils import get_training_utilization, get_training_utilization_details

CPU_MODEL = "aqsol-regression"  # ml.m5.xlarge, from test_artifacts/create_aqsol_artifacts.py
HPO_MODEL = "aqsol-chemprop-hpo"  # ml.g6.12xlarge, one job covering the whole search

RETENTION_DAYS = 15


def _utilization_or_skip(model_name: str) -> pd.DataFrame:
    """The model's utilization, skipping when the fixture or its datapoints are gone."""
    model = Model(model_name)
    if not model.exists():
        pytest.skip(f"{model_name} is not in this account")

    job_name = model.training_job_name
    assert job_name is not None, f"{model_name} should have a training job"

    end_time = model.boto3_session.client("sagemaker").describe_training_job(TrainingJobName=job_name)[
        "TrainingEndTime"
    ]
    age = datetime.now(timezone.utc) - end_time
    if age > timedelta(days=RETENTION_DAYS):
        pytest.skip(f"{job_name} ended {age.days} days ago, past CloudWatch's {RETENTION_DAYS}-day retention")

    df = get_training_utilization_details(model)
    assert df is not None, f"{job_name} is within retention so it should have datapoints"
    return df


@pytest.mark.medium
def test_cpu_job_shape():
    """A CPU box reports cpu/memory/disk, and gets a per-core CPU column."""
    df = _utilization_or_skip(CPU_MODEL)

    assert df.index.name == "timestamp"
    assert df.index.is_monotonic_increasing
    assert {"cpu", "memory", "disk", "cpu_per_core"} <= set(df.columns)

    # SageMaker publishes no GPU metrics for a CPU instance, so those columns never appear
    assert not [column for column in df.columns if column.startswith("gpu")]

    assert df.attrs["instance_count"] == 1
    assert df.attrs["num_gpus"] == 0
    assert df.attrs["training_job_name"].startswith(CPU_MODEL)


@pytest.mark.medium
def test_cpu_per_core_divides_by_core_count():
    """cpu is summed across cores; cpu_per_core is that sum over the instance's core count."""
    df = _utilization_or_skip(CPU_MODEL)
    num_cpus = df.attrs["num_cpus"]

    assert num_cpus > 0
    pd.testing.assert_series_equal(df["cpu"] / num_cpus, df["cpu_per_core"], check_names=False)
    assert df["cpu_per_core"].max() <= 100.0


@pytest.mark.medium
def test_hpo_job_spans_the_whole_search():
    """An HPO run is one training job, so its utilization covers the search end to end."""
    df = _utilization_or_skip(HPO_MODEL)

    assert len(df) > 1, "a search should run long enough for several 1-minute datapoints"
    assert df.attrs["training_job_name"].startswith(HPO_MODEL)
    assert df["cpu"].max() > 0


@pytest.mark.medium
def test_gpu_job_gets_per_device_column():
    """A GPU box reports GPU metrics, and gets a per-device GPU column."""
    df = _utilization_or_skip(HPO_MODEL)
    num_gpus = df.attrs["num_gpus"]

    assert num_gpus > 0, f"{HPO_MODEL} should be on a GPU instance"
    pd.testing.assert_series_equal(df["gpu"] / num_gpus, df["gpu_per_device"], check_names=False)

    # gpu_memory is already a percentage of the device, so it is left undivided
    assert "gpu_memory" in df.columns
    assert "gpu_memory_per_device" not in df.columns


@pytest.mark.medium
def test_summary_rows_and_hardware():
    """The summary carries a row per metric, with the hardware on the index name."""
    model = Model(HPO_MODEL)
    if not model.exists():
        pytest.skip(f"{HPO_MODEL} is not in this account")
    summary = get_training_utilization(model)
    if summary is None:
        pytest.skip(f"{HPO_MODEL} is past CloudWatch's {RETENTION_DAYS}-day retention")

    assert list(summary.columns) == ["mean", "median", "peak"]
    assert {"cpu", "cpu_per_core", "gpu", "gpu_per_device", "memory"} <= set(summary.index)

    # The hardware is on the index name because that is what pandas actually prints
    assert summary.attrs["instance_type"] in summary.index.name
    assert f"{summary.attrs['num_gpus']} GPUs" in summary.index.name


def test_model_copy_has_no_training_job():
    """A promoted copy holds a frozen artifact and never ran a job of its own."""

    class ModelCopy:
        name = "promoted-copy"
        training_job_name = None

    assert get_training_utilization_details(ModelCopy()) is None
    assert get_training_utilization(ModelCopy()) is None
