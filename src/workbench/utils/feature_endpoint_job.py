"""Run a feature endpoint over a DataFrame as a tracked subprocess.

A featurization pass over thousands of molecules takes long enough that running it
inline freezes the caller's session with no progress and no way to redirect. This
runs it as a subprocess with the same ``wait=False`` semantics and job tracking as
local model training, so the caller gets a handle back immediately.

The cache key is chosen here rather than by the caller. Standardizing endpoints
rewrite the ``smiles`` column and preserve the input as ``orig_smiles``, so a cache
keyed on the input column misses nearly everything on the next run.

Usage:
    ```python
    job = run_feature_endpoint(df, "smiles-to-3d-v2", name="cyp_3d", wait=False)
    job.state()      # {"state": "running", ...}
    job.results()    # DataFrame once the state is "completed"
    ```
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Union

import pandas as pd

from workbench.local import storage
from workbench.utils import job_tracker

log = logging.getLogger("workbench")

# The column standardizing endpoints preserve the caller's input key in
DEFAULT_OUTPUT_KEY = "orig_smiles"


def job_root(create: bool = False) -> str:
    """The directory holding featurization job state.

    Args:
        create (bool): Create it if missing (default: False)

    Returns:
        str: Path to the featurization jobs directory
    """
    path = os.path.join(storage.local_root(create=create), "feature_endpoint_jobs")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


class FeatureEndpointJob:
    """A featurization pass, running or finished.

    Attributes:
        name (str): The job name
        path (str): The directory holding this job's input, output, and status
    """

    def __init__(self, name: str):
        """Initialize a FeatureEndpointJob

        Args:
            name (str): The job name
        """
        self.name = name
        self.path = os.path.join(job_root(), name)
        self.input_path = os.path.join(self.path, "input.parquet")
        self.output_path = os.path.join(self.path, "output.parquet")
        self.status_path = os.path.join(self.path, "status.json")
        self.log_path = os.path.join(self.path, "job.log")

    def state(self) -> dict:
        """The job's current state.

        Returns:
            dict: {state, rows, endpoint, ...}, where state is one of
                "running", "completed", "failed", or "interrupted"
        """
        if not os.path.isfile(self.status_path):
            return {"state": "unknown"}
        with open(self.status_path, "r") as fp:
            status = json.load(fp)

        # A watcher still polling owes this job an outcome, so a dead child means the
        # result is pending, not lost. With nobody watching, the results file -- the
        # child's last write -- separates a finished run from one that died partway
        if status.get("state") == "running" and not _pid_alive(status.get("pid")):
            if not job_tracker.is_watched(self.name):
                status["state"] = "completed" if os.path.isfile(self.output_path) else "interrupted"
        return status

    def results(self) -> Union[pd.DataFrame, None]:
        """The featurized rows.

        Returns:
            Union[pd.DataFrame, None]: The results, or None if the job hasn't finished
        """
        if not os.path.isfile(self.output_path):
            return None
        return pd.read_parquet(self.output_path)

    def job_log(self, lines: int = None) -> str:
        """The subprocess output.

        Args:
            lines (int, optional): Only the last N lines

        Returns:
            str: The log contents (empty if the job never started)
        """
        if not os.path.isfile(self.log_path):
            return ""
        with open(self.log_path, "r") as fp:
            content = fp.readlines()
        return "".join(content[-lines:] if lines else content)

    def _write_status(self, **fields):
        """Internal: Merge fields into the status file"""
        status = {}
        if os.path.isfile(self.status_path):
            with open(self.status_path, "r") as fp:
                status = json.load(fp)
        status.update(fields)
        with open(self.status_path, "w") as fp:
            json.dump(status, fp, indent=4, default=str)

    def _record_outcome(self, returncode: int) -> bool:
        """Internal: Write the durable outcome, for both the blocking and tracked paths

        Args:
            returncode (int): The subprocess exit code

        Returns:
            bool: True if the pass succeeded
        """
        success = returncode == 0 and os.path.isfile(self.output_path)
        self._write_status(state="completed" if success else "failed", returncode=returncode)
        return success


def _pid_alive(pid) -> bool:
    """Internal: Is this pid still running?"""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def run_feature_endpoint(
    eval_df: pd.DataFrame,
    endpoint: str,
    name: str,
    cache_key_column: str = "smiles",
    output_key_column: Union[str, None] = DEFAULT_OUTPUT_KEY,
    wait: bool = True,
) -> FeatureEndpointJob:
    """Run a feature endpoint over a DataFrame as a subprocess.

    Args:
        eval_df (pd.DataFrame): The rows to featurize
        endpoint (str): Name of the feature endpoint
        name (str): Name for this job, used for its directory and job tracking
        cache_key_column (str): Column whose values key the cache (default: "smiles")
        output_key_column (str, optional): Column in the endpoint's output holding the
            caller's original key. Standardizing endpoints rewrite the key column and
            preserve the input here, so keying on it is what keeps cache hits across
            runs. Pass None for an endpoint that leaves the key column alone.
        wait (bool): Block until the pass finishes, streaming output (default: True)

    Returns:
        FeatureEndpointJob: The job, finished when wait=True
    """
    job = FeatureEndpointJob(name)
    os.makedirs(job.path, exist_ok=True)
    eval_df.to_parquet(job.input_path, index=False)

    command = [
        sys.executable,
        "-m",
        "workbench.utils.feature_endpoint_job",
        "--endpoint",
        endpoint,
        "--input",
        job.input_path,
        "--output",
        job.output_path,
        "--cache-key-column",
        cache_key_column,
    ]
    if output_key_column:
        command += ["--output-key-column", output_key_column]

    log.important(f"Featurizing {len(eval_df)} rows through {endpoint} as job '{name}'...")
    job._write_status(state="running", endpoint=endpoint, rows=len(eval_df))

    if not wait:
        # The child dups the descriptor, so ours is closed as soon as it's launched
        with open(job.log_path, "w") as log_file:
            proc = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        job._write_status(pid=proc.pid)
        job_tracker.watch_subprocess(
            name,
            proc,
            kind="Featurization",
            log_path=job.log_path,
            on_finish=job._record_outcome,
        )
        return job

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    job._write_status(pid=proc.pid)

    # Stream the child's output so a long pass isn't a silent block
    with open(job.log_path, "w") as log_file:
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
    proc.wait()

    if not job._record_outcome(proc.returncode):
        raise RuntimeError(f"Featurization job '{name}' failed:\n{job.job_log(20)}")
    return job


def main():
    """Run one featurization pass. This is the subprocess entry point."""
    parser = argparse.ArgumentParser(description="Run a feature endpoint over a parquet file")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-key-column", default="smiles")
    parser.add_argument("--output-key-column", default=None)
    args = parser.parse_args()

    from workbench.api import Endpoint
    from workbench.api.inference_cache import InferenceCache

    eval_df = pd.read_parquet(args.input)
    cache = InferenceCache(
        Endpoint(args.endpoint),
        cache_key_column=args.cache_key_column,
        output_key_column=args.output_key_column,
    )
    log.important(f"Cache holds {cache.cache_size()} rows before this pass...")

    results = cache.inference(eval_df)

    # A missing output key column means every row was cached under the wrong value,
    # so the next run would recompute all of it. Fail rather than bank the work.
    if args.output_key_column and args.output_key_column not in results.columns:
        raise ValueError(
            f"Endpoint {args.endpoint} returned no '{args.output_key_column}' column, "
            "so the cache is keyed on values the next run won't look up. Pass "
            "output_key_column=None if this endpoint doesn't rewrite the key column."
        )

    results.to_parquet(args.output, index=False)
    log.important(f"Featurized {len(results)} rows, cache now holds {cache.cache_size()}...")


if __name__ == "__main__":
    main()
