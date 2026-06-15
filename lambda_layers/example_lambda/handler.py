"""Example AWS Lambda exercising the workbench Lambda layer.

A *read-only* stand-in for the client's nightly DT lambda: it builds a
``PipelineManager`` from an ml_pipelines prefix, computes the plan, and returns it
as JSON. It deliberately does NOT submit anything to SQS -- the point is to prove
the layer imports and the graph/plan logic runs in the real Lambda runtime (its
boto3 + the layer's linux wheels) *before* we touch the client's lambda.

Invoke with an event:
    {"pipelines_path": "s3://my-bucket/ml_pipelines/"}     # real AWS mtimes
    {"pipelines_path": "s3://...", "simulate_modified": ["ds:foo"]}   # offline sim
Or set ML_PIPELINES_BUCKET and invoke with {} (uses s3://$BUCKET/ml_pipelines/).
"""

import json
import os

import workbench
from workbench.lambda_layer.pipeline_manager import PipelineManager, simulated_mtime


def lambda_handler(event, context=None):
    event = event or {}

    # Where the pipelines.json files live (an s3:// prefix in Lambda; a local dir
    # works too, which is what the offline test uses).
    path = event.get("pipelines_path")
    if not path:
        path = f"s3://{os.environ['ML_PIPELINES_BUCKET']}/ml_pipelines/"

    print(f"workbench {workbench.__version__} -- planning from {path}")
    pm = PipelineManager(path)  # default boto3: Lambda role + region

    # Real AWS mtimes by default; an optional simulate list makes the run offline
    # (and demonstrates forward-flood propagation without touching AWS).
    mtime_fn = simulated_mtime(event["simulate_modified"]) if "simulate_modified" in event else None
    decisions = pm.plan(mtime_fn)

    plan = [{"job": d.job.node_id, "group": d.job.group, "run": d.run, "reason": d.reason} for d in decisions]
    will_run = [p["job"] for p in plan if p["run"]]
    print(f"{len(will_run)}/{len(plan)} jobs would run (suspect_environment={pm.suspect_environment})")

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "workbench_version": workbench.__version__,
                "pipelines_path": path,
                "num_pipelines": pm.get_num_pipelines(),
                "suspect_environment": pm.suspect_environment,
                "will_run": will_run,
                "plan": plan,
            },
            indent=2,
        ),
    }
