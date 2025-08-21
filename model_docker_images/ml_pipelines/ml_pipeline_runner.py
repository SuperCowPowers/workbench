import os
import sys
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

    # Make it executable
    os.chmod(local_path, 0o755)
    log.info(f"Downloaded and made executable: {local_path}")


def run_ml_pipeline(script_path: str):
    """Execute the ML pipeline script."""
    log.info(f"Executing ML pipeline: {script_path}")

    try:
        # Run the script with python, streaming output in real-time
        process = subprocess.Popen(
            [sys.executable, "-u", script_path],  # -u for unbuffered Python output
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
        )

        # Stream output line by line
        for line in process.stdout:
            print(line, end="", flush=True)  # Print and flush immediately

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            log.info("ML pipeline completed successfully")
        else:
            log.error(f"ML pipeline failed with exit code {return_code}")
        return return_code

    except Exception as e:
        log.error(f"ML pipeline execution error: {e}")
        return 1


def main():
    # Report the version of the workbench package
    log.info(f"Workbench version: {workbench.__version__}")

    # Get the ML Pipeline S3 path from environment variable
    ml_pipeline_s3_path = os.environ.get("ML_PIPELINE_S3_PATH")

    if not ml_pipeline_s3_path:
        log.error("ML_PIPELINE_S3_PATH environment variable not set")
        sys.exit(1)

    # Extract script name for local file
    script_name = os.path.basename(ml_pipeline_s3_path)
    local_script_path = f"/tmp/{script_name}"

    try:
        # Download the ML pipeline script from S3
        download_ml_pipeline_from_s3(ml_pipeline_s3_path, local_script_path)

        # Execute the ML pipeline
        exit_code = run_ml_pipeline(local_script_path)

        # Clean up
        os.remove(local_script_path)

        sys.exit(exit_code)

    except Exception as e:
        log.error(f"Error in run_script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
