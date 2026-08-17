"""Tests for feature endpoint jobs (no AWS required)

The subprocess is stubbed rather than launched, so these cover what the parent owns:
the job's on-disk state, how a finished-but-not-yet-recorded run reads, and that a
reused name doesn't serve the previous run's results. The child's own pass is covered
separately with a stubbed InferenceCache.
"""

import json
import os
import subprocess
import sys
import time
import types

import pandas as pd
import pytest

from workbench.utils import job_tracker
from workbench.utils.feature_endpoint_job import FeatureEndpointJob, job_root, main, run_feature_endpoint
from workbench.utils.config_manager import ConfigManager


@pytest.fixture(autouse=True)
def local_storage(tmp_path):
    """Point local storage at a temp directory for every test"""
    cm = ConfigManager()
    original = cm.config.get("WORKBENCH_LOCAL_PATH")
    cm.set_config("WORKBENCH_LOCAL_PATH", str(tmp_path))
    yield tmp_path
    cm.set_config("WORKBENCH_LOCAL_PATH", original)


@pytest.fixture(autouse=True)
def clean_tracker():
    """Each test starts with an empty job tracker"""
    job_tracker._watched.clear()
    job_tracker.drain_completed()
    yield
    job_tracker._watched.clear()
    job_tracker.drain_completed()


@pytest.fixture
def eval_df():
    return pd.DataFrame({"id": [1, 2, 3], "smiles": ["CCO", "CCC", "CCN"]})


# Bound before any test patches subprocess.Popen, so dead_pid() still spawns a real
# process when the fake is installed
_REAL_POPEN = subprocess.Popen


def dead_pid() -> int:
    """A pid that has certainly exited"""
    proc = _REAL_POPEN([sys.executable, "-c", ""])
    proc.wait()
    return proc.pid


def fake_popen(returncode: int = 0, writes_output: bool = True):
    """A Popen stand-in that optionally writes the child's output file"""

    class FakePopen:
        def __init__(self, command, **kwargs):
            self.returncode = returncode
            self.pid = dead_pid()
            self.stdout = ["featurizing...\n", "done\n"]
            output = command[command.index("--output") + 1]
            if writes_output:
                pd.DataFrame({"smiles": ["CCO"], "orig_smiles": ["CCO"], "molwt": [46.07]}).to_parquet(
                    output, index=False
                )

        def poll(self):
            """Already exited -- the watcher reports on its first check"""
            return self.returncode

        def wait(self):
            return self.returncode

    return FakePopen


class TestJobState:
    def test_no_run_is_unknown(self):
        assert FeatureEndpointJob("never-ran").state()["state"] == "unknown"

    def test_pending_while_a_watcher_still_owes_an_outcome(self, eval_df):
        """The watcher polls on an interval, so a finished child isn't yet recorded"""
        job = FeatureEndpointJob("pending")
        job._init_dirs()
        job._write_status(state="running", pid=dead_pid())
        job_tracker.register("pending")

        assert job.state()["state"] == "running"

    def test_completed_when_the_results_are_on_disk(self, eval_df):
        job = FeatureEndpointJob("finished")
        job._init_dirs()
        job._write_status(state="running", pid=dead_pid())
        eval_df.to_parquet(job.output_path, index=False)

        assert job.state()["state"] == "completed"

    def test_interrupted_when_the_run_died_partway(self):
        """Nobody watching and no results means the work was lost, not pending"""
        job = FeatureEndpointJob("orphan")
        job._init_dirs()
        job._write_status(state="running", pid=dead_pid())

        assert job.state()["state"] == "interrupted"

    def test_recorded_outcome_wins_over_the_pid(self, eval_df):
        job = FeatureEndpointJob("recorded")
        job._init_dirs()
        job._write_status(state="running", pid=dead_pid())
        eval_df.to_parquet(job.output_path, index=False)
        job._record_outcome(0)

        assert job.state()["state"] == "completed"
        assert job.state()["returncode"] == 0


class TestOutcome:
    def test_nonzero_exit_is_a_failure(self):
        job = FeatureEndpointJob("bad-exit")
        job._init_dirs()
        assert job._record_outcome(1) is False
        assert job.state()["state"] == "failed"

    def test_clean_exit_without_results_is_a_failure(self):
        """The child writes its results last, so a clean exit with none didn't finish"""
        job = FeatureEndpointJob("no-output")
        job._init_dirs()
        assert job._record_outcome(0) is False

    def test_results_are_none_before_the_run_finishes(self):
        job = FeatureEndpointJob("unfinished")
        job._init_dirs()
        assert job.results() is None

    def test_job_log_is_empty_before_the_run_starts(self):
        assert FeatureEndpointJob("no-log").job_log() == ""

    def test_job_log_tails(self):
        job = FeatureEndpointJob("logged")
        job._init_dirs()
        with open(job.log_path, "w") as fp:
            fp.write("one\ntwo\nthree\n")
        assert job.job_log(2) == "two\nthree\n"


class TestInitDirs:
    def test_reused_name_starts_empty(self, eval_df):
        """A previous run's results must not be served as this run's"""
        job = FeatureEndpointJob("reused")
        job._init_dirs()
        eval_df.to_parquet(job.output_path, index=False)
        job._write_status(state="completed")

        FeatureEndpointJob("reused")._init_dirs()
        assert not os.path.isfile(job.output_path)
        assert not os.path.isfile(job.status_path)

    def test_jobs_live_outside_the_artifact_directories(self):
        """LocalMeta globs the artifact subdirs, so jobs must not look like artifacts"""
        assert os.path.basename(job_root()) == "feature_endpoint_jobs"


class TestLaunch:
    def test_detached_run_records_what_it_launched(self, eval_df, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", fake_popen())
        job = run_feature_endpoint(eval_df, "smiles-to-2d-v1", name="detached", wait=False)

        status = json.load(open(job.status_path))
        assert status["endpoint"] == "smiles-to-2d-v1"
        assert status["rows"] == 3

    def test_detached_run_reaches_its_outcome_through_the_watcher(self, eval_df, monkeypatch):
        """The tracked path must leave the same on-disk state as the blocking one"""
        monkeypatch.setattr(subprocess, "Popen", fake_popen())
        job = run_feature_endpoint(eval_df, "smiles-to-2d-v1", name="watched", wait=False)

        deadline = time.time() + 30
        while time.time() < deadline and job.state()["state"] == "running":
            time.sleep(0.1)

        assert job.state()["state"] == "completed"
        assert job.state()["returncode"] == 0

    def test_blocking_run_records_its_outcome(self, eval_df, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", fake_popen())
        job = run_feature_endpoint(eval_df, "smiles-to-2d-v1", name="blocking")

        assert job.state()["state"] == "completed"
        assert len(job.results()) == 1

    def test_blocking_run_raises_with_the_log_tail(self, eval_df, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", fake_popen(returncode=1, writes_output=False))

        with pytest.raises(RuntimeError, match="featurizing"):
            run_feature_endpoint(eval_df, "smiles-to-2d-v1", name="doomed")

    def test_input_is_staged_for_the_child(self, eval_df, monkeypatch):
        monkeypatch.setattr(subprocess, "Popen", fake_popen())
        job = run_feature_endpoint(eval_df, "smiles-to-2d-v1", name="staged", wait=False)

        assert pd.read_parquet(job.input_path).equals(eval_df)


class TestChildPass:
    """The child's own run, with the endpoint and its cache stubbed out"""

    def test_writes_the_cached_results(self, eval_df, monkeypatch, tmp_path):
        captured = {}

        class FakeCache:
            def __init__(self, endpoint, cache_key_column="smiles"):
                captured["endpoint"] = endpoint
                captured["cache_key_column"] = cache_key_column

            def cache_size(self):
                return 0

            def inference(self, df):
                return df.assign(molwt=1.0)

        fake_api = types.ModuleType("workbench.api")
        fake_api.Endpoint = lambda name: name
        fake_cache_module = types.ModuleType("workbench.api.inference_cache")
        fake_cache_module.InferenceCache = FakeCache
        monkeypatch.setitem(sys.modules, "workbench.api", fake_api)
        monkeypatch.setitem(sys.modules, "workbench.api.inference_cache", fake_cache_module)

        input_path = str(tmp_path / "in.parquet")
        output_path = str(tmp_path / "out.parquet")
        eval_df.to_parquet(input_path, index=False)
        monkeypatch.setattr(
            sys, "argv", ["featurize", "--endpoint", "smiles-to-2d-v1", "--input", input_path, "--output", output_path]
        )

        main()

        # The cache is constructed with the endpoint's defaults, not per-caller arguments
        assert captured == {"endpoint": "smiles-to-2d-v1", "cache_key_column": "smiles"}
        assert "molwt" in pd.read_parquet(output_path).columns
