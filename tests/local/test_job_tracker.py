"""Tests for the shared job tracker (no AWS required)"""

import subprocess
import sys
import time

import pytest

from workbench.utils import job_tracker
from workbench.utils.bosco_utils import job_updates


@pytest.fixture(autouse=True)
def clean_tracker():
    """Each test starts with an empty tracker"""
    job_tracker.drain_completed()
    job_tracker._watched.clear()
    yield
    job_tracker.drain_completed()
    job_tracker._watched.clear()


def run_python(code: str) -> subprocess.Popen:
    """Start a child python process running the given source"""
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def wait_for_completion(timeout: float = 30.0) -> list:
    """Poll drain_completed until a row shows up (or give up)"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = job_tracker.drain_completed()
        if rows:
            return rows
        time.sleep(0.1)
    return []


class TestSubprocessWatcher:
    def test_success_reported(self):
        proc = run_python("print('done')")
        job_tracker.watch_subprocess("happy_job", proc, kind="Local training", interval=1)

        rows = wait_for_completion()
        assert len(rows) == 1
        assert rows[0]["name"] == "happy_job"
        assert rows[0]["status"] == "COMPLETED"
        assert rows[0]["kind"] == "Local training"
        assert job_tracker._watched["happy_job"] == "completed"

    def test_failure_reported_with_exit_code(self):
        proc = run_python("import sys; sys.exit(3)")
        job_tracker.watch_subprocess("sad_job", proc, kind="Local training", interval=1)

        rows = wait_for_completion()
        assert rows[0]["status"] == "FAILED"
        assert "exit 3" in rows[0]["reason"]
        assert job_tracker._watched["sad_job"] == "failed"

    def test_failure_reason_includes_log_tail(self, tmp_path):
        log_path = tmp_path / "train.log"
        log_path.write_text("epoch 1\nepoch 2\nValueError: bad target column\n")

        proc = run_python("import sys; sys.exit(1)")
        job_tracker.watch_subprocess("logged_job", proc, log_path=str(log_path), interval=1)

        rows = wait_for_completion()
        assert "bad target column" in rows[0]["reason"]

    def test_missing_log_still_reports(self, tmp_path):
        proc = run_python("import sys; sys.exit(1)")
        job_tracker.watch_subprocess("no_log", proc, log_path=str(tmp_path / "nope.log"), interval=1)

        rows = wait_for_completion()
        assert rows[0]["reason"] == "exit 1"


class TestLights:
    def test_no_jobs_no_lights(self):
        assert job_tracker.job_lights() == []

    def test_running_and_finished_states(self):
        job_tracker.register("job_a")
        assert len(job_tracker.job_lights()) == 3  # "Jobs [", one dot, "]"

        job_tracker.report({"name": "job_a", "status": "COMPLETED"}, success=True)
        assert job_tracker._watched["job_a"] == "completed"


class TestJobUpdates:
    def test_prefixes_prompt_per_kind(self):
        job_tracker.report({"kind": "Batch job", "name": "sweep", "status": "SUCCEEDED"}, success=True)
        job_tracker.report(
            {"kind": "Local training", "name": "aqsol", "status": "FAILED", "reason": "exit 1"}, success=False
        )

        out = job_updates("what happened?")
        assert "[Batch job update: sweep SUCCEEDED]" in out
        assert "[Local training update: aqsol FAILED -- exit 1]" in out
        assert out.endswith("what happened?")

    def test_no_jobs_passes_prompt_through(self):
        assert job_updates("hello") == "hello"

    def test_drain_is_one_shot(self):
        job_tracker.report({"kind": "Batch job", "name": "sweep", "status": "SUCCEEDED"}, success=True)
        assert "sweep" in job_updates("first")
        assert job_updates("second") == "second"
