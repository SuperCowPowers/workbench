"""Can an Atom-JEPA async endpoint fall back to a CPU instance pool?

A variant has one host image for all its pools, and GPU pools need the GPU one. This deploys a
CPU pool first (so it lands there without waiting on GPU capacity) and a GPU pool second, under
that GPU host image, then scores a few molecules on whatever instance came up. Everything it
creates is deleted at the end unless --keep is passed.

    python spikes/atomjepa_cpu_timing/cpu_pool_test.py --model cyp-reg-atomjepa-3a4-readout --molecules 5
"""

import argparse
import time

from botocore.exceptions import WaiterError

from workbench.api import FeatureSet, Model
from workbench.core.cloud_platform.aws.aws_account_clamp import AWSAccountClamp
from workbench.core.transforms.model_to_endpoint.model_to_endpoint import GPU_INFERENCE_AMI
from workbench.endpoints.async_inference import async_inference
from workbench.utils.config_manager import ConfigManager

NAME = "atomjepa-cpu-pool-test"
POOLS = [{"InstanceType": "ml.c7i.4xlarge", "Priority": 1}, {"InstanceType": "ml.g6.xlarge", "Priority": 2}]


def report(sm, start: float) -> str:
    """Print the endpoint's status, failure reason, and serving pool; return the status"""
    desc = sm.describe_endpoint(EndpointName=NAME)
    variant = (desc.get("ProductionVariants") or [{}])[0]
    serving = [p["InstanceType"] for p in variant.get("InstancePools", []) if p.get("CurrentInstanceCount")]
    print(f"\nStatus: {desc['EndpointStatus']} after {(time.time() - start) / 60:.1f} min")
    print(f"Serving pool: {serving or '-'}")
    if desc.get("FailureReason"):
        print(f"Failure reason: {desc['FailureReason']}")
    return desc["EndpointStatus"]


def delete(sm):
    """Delete the endpoint, its config, and its model"""
    # SageMaker can't delete an endpoint while it is creating, so wait that out first
    try:
        status = sm.describe_endpoint(EndpointName=NAME)["EndpointStatus"]
        if status in ("Creating", "Updating"):
            print(f"{NAME} is {status}; waiting for that to finish before deleting it...")
        while status in ("Creating", "Updating"):
            time.sleep(30)
            status = sm.describe_endpoint(EndpointName=NAME)["EndpointStatus"]
    except sm.exceptions.ClientError:
        pass  # no endpoint to wait on
    for call, kwargs in [
        (sm.delete_endpoint, {"EndpointName": NAME}),
        (sm.delete_endpoint_config, {"EndpointConfigName": NAME}),
        (sm.delete_model, {"ModelName": NAME}),
    ]:
        try:
            call(**kwargs)
        except sm.exceptions.ClientError as e:
            print(f"{call.__name__}: {e.response['Error']['Message']}")
    print(f"Deleted {NAME}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="cyp-reg-atomjepa-3a4-readout")
    parser.add_argument("--molecules", type=int, default=5)
    parser.add_argument("--keep", action="store_true", help="leave the endpoint up afterward")
    args = parser.parse_args()

    clamp = AWSAccountClamp()
    sm = clamp.boto3_session.client("sagemaker")
    bucket = ConfigManager().get_config("WORKBENCH_BUCKET")
    model = Model(args.model)

    sm.create_model(
        ModelName=NAME,
        PrimaryContainer={"ModelPackageName": model.model_package_arn()},
        ExecutionRoleArn=clamp.aws_session.get_workbench_execution_role_arn(),
    )
    try:
        sm.create_endpoint_config(
            EndpointConfigName=NAME,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": NAME,
                    "InitialInstanceCount": 1,
                    "InstancePools": POOLS,
                    "VariantInstanceProvisionTimeoutInSeconds": 600,
                    "InferenceAmiVersion": GPU_INFERENCE_AMI,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 300,
                }
            ],
            AsyncInferenceConfig={
                "OutputConfig": {
                    "S3OutputPath": f"s3://{bucket}/endpoints/{NAME}/async-output",
                    "S3FailurePath": f"s3://{bucket}/endpoints/{NAME}/async-failures",
                }
            },
        )
        sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
        print(f"Creating {NAME}: pools {[p['InstanceType'] for p in POOLS]}, host image {GPU_INFERENCE_AMI}...")
        start = time.time()
        try:
            sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME, WaiterConfig={"Delay": 30, "MaxAttempts": 80})
        except WaiterError:
            pass  # report() shows why
        if report(sm, start) != "InService":
            return

        # Score a few molecules on whichever instance came up
        smiles_column, *extra_features = model.features()
        df = FeatureSet(model.get_input()).pull_dataframe().sample(args.molecules, random_state=0)
        start = time.time()
        result = async_inference(
            NAME,
            df[[smiles_column, *extra_features]],
            sm_session=clamp.boto3_session,
            batch_size=args.molecules,
            s3_bucket=bucket,
        )
        print(result.head(args.molecules).to_string())
        print(f"\n{len(df)} molecules scored in {time.time() - start:.0f}s")
    finally:
        if not args.keep:
            delete(sm)


if __name__ == "__main__":
    main()
