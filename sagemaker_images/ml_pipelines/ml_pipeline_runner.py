import os
import sys
import json
import subprocess
import boto3
import logging
from urllib.parse import urlparse
import workbench

# Set up logging
log = logging.getLogger("workbench")


def download_ml_pipeline_from_s3(s3_path: str, local_path: str):
    """Download ML Pipeline from S3 to local filesystem."""
    parsed = urlparse(s3_path)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    log.info(f"Downloading {s3_path} to {local_path}")
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, local_path)


def resolve_script_ref(ref: str) -> str:
    """Resolve a pipeline script ref to a local runnable path, dispatched by scheme.

        workbench:<path> -> a script bundled in the workbench package (run in place)
        plugin:<path>    -> a script under WORKBENCH_PLUGINS (local, or copied from s3)
        s3://... / path  -> downloaded from S3 to /tmp (the default)

    Mirrors the schemes the PipelineManager resolver passes through unchanged.
    """
    if ref.startswith("workbench:"):
        base = os.path.join(os.path.dirname(workbench.__file__), "batch")
        path = os.path.join(base, ref[len("workbench:") :])
        if not os.path.exists(path):
            raise FileNotFoundError(f"workbench script not found: {path}")
        return path

    if ref.startswith("plugin:"):
        from workbench.utils.config_manager import ConfigManager

        plugin_dir = ConfigManager().get_config("WORKBENCH_PLUGINS")
        if not plugin_dir:
            raise RuntimeError("plugin: script requires WORKBENCH_PLUGINS to be set")
        src = f"{plugin_dir.rstrip('/')}/{ref[len('plugin:'):]}"
        if src.startswith("s3://"):
            local = f"/tmp/{os.path.basename(src)}"
            download_ml_pipeline_from_s3(src, local)
            return local
        return src

    # Default: a full S3 URI -> download to /tmp
    local = f"/tmp/{os.path.basename(ref)}"
    download_ml_pipeline_from_s3(ref, local)
    return local


def run_ml_pipeline(script_path: str, script_args: list[str] | None = None):
    """Execute the ML pipeline script.

    Args:
        script_path (str): Local path to the downloaded pipeline script
        script_args (list[str] | None): Args forwarded verbatim to the script
    """
    cmd = [sys.executable, script_path, *(script_args or [])]
    log.info(f"Executing ML pipeline: {' '.join(cmd)}")
    try:
        # Run the script with python (don't raise on non-zero exit)
        result = subprocess.run(cmd, check=False, text=True)
        if result.returncode == 0:
            log.info("ML pipeline completed successfully")
        else:
            log.error(f"ML pipeline failed with exit code {result.returncode}")

        return result.returncode
    except Exception as e:
        log.error(f"ML pipeline execution error: {e}")
        return 1


def main():
    """Main entry point for the ML pipeline runner."""

    # Report the version of the workbench package
    log.info(f"Workbench version: {workbench.__version__}")

    # The pipeline script ref (env name kept for compatibility; value may now carry a
    # scheme -- workbench:/plugin:/s3:// -- resolved by resolve_script_ref).
    script_ref = os.environ.get("ML_PIPELINE_S3_PATH")
    if not script_ref:
        log.error("ML_PIPELINE_S3_PATH environment variable not set")
        sys.exit(1)

    # Args to forward to the pipeline script (JSON-encoded list, if any)
    script_args = json.loads(os.environ.get("PIPELINE_ARGS", "[]"))

    try:
        # Resolve the script ref to a local path (download only if needed)
        local_script_path = resolve_script_ref(script_ref)

        # Execute the ML pipeline
        exit_code = run_ml_pipeline(local_script_path, script_args)

        # Clean up only what we downloaded (in-package/plugin scripts run in place)
        if local_script_path.startswith("/tmp/"):
            os.remove(local_script_path)
        sys.exit(exit_code)
    except Exception as e:
        log.error(f"Error in run_script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
