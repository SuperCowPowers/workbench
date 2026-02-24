"""Batch Trigger Lambda: Processes SQS messages and submits jobs to AWS Batch.

This Lambda handles job dependencies by:
1. Reading outputs/inputs from the SQS message body
2. Querying Batch for active jobs that produce what this job needs
3. Submitting with dependsOn to ensure proper execution order
"""

import json
import os
import boto3
from datetime import datetime

batch = boto3.client("batch")

WORKBENCH_BUCKET = os.environ["WORKBENCH_BUCKET"]
JOB_QUEUE = os.environ["JOB_QUEUE"]
JOB_DEFINITIONS = {
    "small": os.environ["JOB_DEF_SMALL"],
    "medium": os.environ["JOB_DEF_MEDIUM"],
    "large": os.environ["JOB_DEF_LARGE"],
}


def find_active_jobs_with_output(output_name: str) -> list[str]:
    """Find all active Batch jobs that produce the specified output.

    Queries for jobs in PENDING, RUNNABLE, STARTING, or RUNNING status
    that have the matching output in their environment.

    Args:
        output_name (str): The output name to look for (e.g., "my_dag:stage_0")

    Returns:
        list[str]: List of job IDs that produce this output
    """
    active_statuses = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]
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
                    # Check container environment for PIPELINE_OUTPUTS
                    env_vars = job.get("container", {}).get("environment", [])
                    for env in env_vars:
                        if env.get("name") == "PIPELINE_OUTPUTS":
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

            # Read dependency info from message
            outputs = message.get("outputs", [])
            inputs = message.get("inputs", [])

            script_name = script_path.rstrip("/").rsplit("/", 1)[-1].removesuffix(".py")
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
                env_vars.append({"name": "PIPELINE_OUTPUTS", "value": ",".join(outputs)})

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
