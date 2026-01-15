"""Batch Trigger Lambda: Processes SQS messages and submits jobs to AWS Batch.

This Lambda handles job dependencies by:
1. Parsing WORKBENCH_BATCH config from the script in S3
2. Translating inputs/outputs to group/priority for dependency tracking
3. Querying Batch for active jobs that produce what this job needs
4. Submitting with dependsOn to ensure proper execution order
"""

import ast
import json
import os
import re
import boto3
from datetime import datetime
from pathlib import Path

batch = boto3.client("batch")
s3 = boto3.client("s3")

WORKBENCH_BUCKET = os.environ["WORKBENCH_BUCKET"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITIONS = {
    "small": os.environ["JOB_DEF_SMALL"],
    "medium": os.environ["JOB_DEF_MEDIUM"],
    "large": os.environ["JOB_DEF_LARGE"],
}


def parse_workbench_batch(script_content: str) -> dict | None:
    """Parse WORKBENCH_BATCH config from a script.

    Looks for a dictionary assignment like:
        WORKBENCH_BATCH = {
            "outputs": ["feature_set_xyz"],
        }
    or:
        WORKBENCH_BATCH = {
            "inputs": ["feature_set_xyz"],
        }

    Args:
        script_content: The Python script content as a string

    Returns:
        The parsed dictionary or None if not found
    """
    pattern = r"WORKBENCH_BATCH\s*=\s*(\{[^}]+\})"
    match = re.search(pattern, script_content, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse WORKBENCH_BATCH: {e}")
            return None
    return None


def translate_to_group_priority(batch_config: dict | None) -> tuple[str | None, int]:
    """Translate inputs/outputs to group/priority.

    - Scripts with outputs are producers (priority 1)
    - Scripts with inputs are consumers (priority 2)
    - Group is derived from the first output or input

    Args:
        batch_config: The parsed WORKBENCH_BATCH config

    Returns:
        Tuple of (group, priority)
    """
    if not batch_config:
        return None, 1

    outputs = batch_config.get("outputs", [])
    inputs = batch_config.get("inputs", [])

    if outputs:
        # Producer: uses first output as group, priority 1
        return outputs[0], 1
    elif inputs:
        # Consumer: uses first input as group, priority 2
        return inputs[0], 2
    else:
        return None, 1


def get_script_content(s3_path: str) -> str:
    """Download and return the content of an S3 script.

    Args:
        s3_path: S3 URI in format s3://bucket/key

    Returns:
        Script content as string
    """
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    key = parts[1]
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read().decode("utf-8")


def find_active_jobs_with_output(output_name: str) -> list[str]:
    """Find all active Batch jobs that produce the specified output.

    Queries for jobs in PENDING, RUNNABLE, STARTING, or RUNNING status
    that have the matching output in their environment.

    Args:
        output_name: The output name to look for (e.g., "feature_set_xyz")

    Returns:
        List of job IDs that produce this output
    """
    active_statuses = ["PENDING", "RUNNABLE", "STARTING", "RUNNING"]
    job_ids = []

    for status in active_statuses:
        try:
            response = batch.list_jobs(jobQueue=JOB_QUEUE, jobStatus=status)
            for job_summary in response.get("jobSummaryList", []):
                job_id = job_summary["jobId"]
                # Get job details to check the outputs
                job_detail = batch.describe_jobs(jobs=[job_id])["jobs"]
                if job_detail:
                    job = job_detail[0]
                    # Check container environment for WORKBENCH_BATCH_OUTPUTS
                    env_vars = job.get("container", {}).get("environment", [])
                    for env in env_vars:
                        if env.get("name") == "WORKBENCH_BATCH_OUTPUTS":
                            job_outputs = env.get("value", "").split(",")
                            if output_name in job_outputs:
                                job_ids.append(job_id)
                                break
        except Exception as e:
            print(f"Warning: Error listing jobs with status {status}: {e}")

    return job_ids


def lambda_handler(event, context):
    """Process SQS messages and submit Batch jobs with dependencies."""

    for record in event["Records"]:
        try:
            message = json.loads(record["body"])
            script_path = message["script_path"]  # s3://bucket/path/to/script.py
            size = message.get("size", "small")
            extra_env = message.get("environment", {})

            # Download script and parse WORKBENCH_BATCH config
            script_content = get_script_content(script_path)
            batch_config = parse_workbench_batch(script_content)

            # Extract inputs/outputs for dependency tracking
            outputs = (batch_config or {}).get("outputs", [])
            inputs = (batch_config or {}).get("inputs", [])

            # Translate to group/priority at the last minute
            group, priority = translate_to_group_priority(batch_config)

            script_name = Path(script_path).stem
            job_name = f"workbench_{script_name}_{datetime.now():%Y%m%d_%H%M%S}"

            # Get job definition name from environment variables
            job_def_name = JOB_DEFINITIONS.get(size, JOB_DEFINITIONS["small"])

            # Build environment variables
            env_vars = [
                {"name": "ML_PIPELINE_S3_PATH", "value": script_path},
                {"name": "WORKBENCH_BUCKET", "value": WORKBENCH_BUCKET},
                *[{"name": k, "value": v} for k, v in extra_env.items()],
            ]

            # Add outputs to environment for dependency tracking (comma-separated)
            if outputs:
                env_vars.append({"name": "WORKBENCH_BATCH_OUTPUTS", "value": ",".join(outputs)})

            # Build job submission parameters
            submit_params = {
                "jobName": job_name,
                "jobQueue": JOB_QUEUE,
                "jobDefinition": job_def_name,
                "containerOverrides": {"environment": env_vars},
            }

            # If this job has inputs, look for jobs that produce those outputs
            if inputs:
                dependency_job_ids = []
                for input_name in inputs:
                    dependency_job_ids.extend(find_active_jobs_with_output(input_name))

                # Deduplicate job IDs
                dependency_job_ids = list(set(dependency_job_ids))

                if dependency_job_ids:
                    # Add dependencies (up to 20 supported by AWS Batch)
                    submit_params["dependsOn"] = [
                        {"jobId": job_id, "type": "SEQUENTIAL"} for job_id in dependency_job_ids[:20]
                    ]
                    print(f"Job {job_name} will depend on {len(dependency_job_ids)} job(s): {dependency_job_ids}")

            # Submit Batch job
            response = batch.submit_job(**submit_params)
            print(f"Submitted job: {job_name} ({response['jobId']})")
            if outputs:
                print(f"  Outputs: {outputs}")
            if inputs:
                print(f"  Inputs: {inputs}")

        except Exception as e:
            print(f"Error processing message: {e}")
            raise  # Let SQS retry via DLQ

    return {"statusCode": 200}
