"""Check each endpoint under a MetaEndpoint for the 'one bad row' case.

A batch whose only row has an unparseable SMILES must come back with the same
columns as a valid row. Endpoints that error or drop columns were trained from an
older model template; the script ends with the ml_pipeline_launcher commands that
retrain them.

The valid row goes first through Endpoint.inference(), which retries serverless
cold starts. The bad row goes through fast_inference(), which surfaces errors
instead of filling NaN.

Usage:
    python scripts/admin/meta_endpoint_bad_row_check.py [meta_endpoint_name]
"""

import sys
from collections import defaultdict

import pandas as pd

from workbench.api import Endpoint
from workbench.api.meta_endpoint import MetaEndpoint
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.lambda_layer.pipeline_manager import PipelineManager, ref_type
from workbench.utils.config_manager import ConfigManager

VALID_ROW = pd.DataFrame({"id": ["valid"], "smiles": ["CC(=O)NC1=CC=C(C=C1)O"]})
BAD_ROW = pd.DataFrame({"id": ["bad"], "smiles": ["not_a_smiles"]})


def check_endpoint(name: str) -> str:
    """Return an empty string if the endpoint handles one bad row, else the problem."""
    end = Endpoint(name)
    try:
        expected = end.inference(VALID_ROW).columns.tolist()
    except Exception as e:
        return f"valid row failed: {str(e)[:150]}"
    try:
        actual = end.fast_inference(BAD_ROW).columns.tolist()
    except Exception as e:
        return f"bad row failed: {str(e)[:150]}"
    missing = [c for c in expected if c not in actual]
    return f"bad row is missing {missing}" if missing else ""


def retrain_jobs(pm: PipelineManager, endpoint_name: str) -> list:
    """The model-training jobs to relaunch so this endpoint picks up the current template."""
    producer = next((j for j in pm.jobs if f"endpoint:{endpoint_name}" in j.outputs), None)
    if producer is None:
        return []
    if any(ref_type(o) == "model" for o in producer.outputs):
        return [producer]
    # A promotion arbiter: its champion is a copy, so retrain the challengers that can win
    challengers = {
        i for i in producer.inputs if ref_type(i) == "model" and "no_promote" not in producer.input_flags.get(i, ())
    }
    return [j for j in pm.jobs if challengers & set(j.outputs)]


def main():
    meta_name = sys.argv[1] if len(sys.argv) > 1 else "all-regression-1"
    dag = MetaEndpoint(meta_name).get_dag()
    input_nodes = set(dag.input_nodes)

    needs_update, skipped = [], []
    for node, name in dag.endpoints.items():
        # Downstream nodes take their parent's output, not SMILES
        if node not in input_nodes:
            skipped.append(name)
            print(f"{name:40} SKIPPED (not an input node)")
            continue
        problem = check_endpoint(name)
        if problem:
            needs_update.append(name)
        print(f"{name:40} {'NEEDS UPDATE: ' + problem if problem else 'OK'}")

    print("=" * 70)
    if skipped:
        print(f"Not checked (downstream nodes): {', '.join(skipped)}")
    if not needs_update:
        print(f"{meta_name}: all {len(dag.endpoints) - len(skipped)} checked endpoints are up to date.")
        return

    print(f"{meta_name}: {len(needs_update)} endpoint(s) need to be updated:")
    for name in needs_update:
        print(f"  {name:40} (model: {Endpoint(name).get_input()})")

    # Map each endpoint to the training jobs that rebuild it
    pm = PipelineManager(ConfigManager().get_config("ML_PIPELINES_ROOT"), session=AWSAccountClamp().boto3_session)
    scripts_by_mode, no_pipeline = defaultdict(set), []
    for name in needs_update:
        jobs = retrain_jobs(pm, name)
        if not jobs:
            no_pipeline.append(name)
        for job in jobs:
            scripts_by_mode[job.mode].add(f"{job.stem}.py")

    if no_pipeline:
        print(f"\nNo pipeline produces: {', '.join(no_pipeline)}")
    if scripts_by_mode:
        print("\nRetrain (from your ml_pipelines directory):")
        for mode, scripts in scripts_by_mode.items():
            flag = f"--{mode} " if mode else ""
            print(f"  ml_pipeline_launcher {flag}{' '.join(sorted(scripts))}")


if __name__ == "__main__":
    main()
