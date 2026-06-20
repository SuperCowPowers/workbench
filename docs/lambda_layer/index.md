!!! tip inline end "Workbench Lambda Layer"
    AWS Lambda is a great way to run lightweight, scheduled Workbench jobs. The Workbench Lambda layer ships a dependency-minimal slice of Workbench you can import directly — no packaging, no container.

The **Workbench Lambda layer** is a dependency-minimal slice of Workbench, published
per region and Python version. It bundles **all of Workbench's source** plus only
`networkx` and `pandas`; `boto3`/`botocore` come from the Lambda runtime. The heavy
dependencies (torch, awswrangler, sagemaker, ...) are intentionally absent, so the
layer stays small and imports fast.

!!! warning inline end "Scope"
    The layer carries the [`workbench.lambda_layer`](https://github.com/SuperCowPowers/workbench/tree/main/src/workbench/lambda_layer) subpackage — today, the `PipelineManager` that powers [ML Pipelines](../ml_pipelines/index.md). It does **not** carry the full Workbench API (e.g. `Meta`/`Model`), which needs the heavy dependencies.

## Published ARNs

Attach the ARN matching your region and Python version (3.12):

**us-east-1**

- `arn:aws:lambda:us-east-1:507740646243:layer:workbench-lambda-layer-us-east-1-python312-wip:3`

**us-west-2**

- `arn:aws:lambda:us-west-2:507740646243:layer:workbench-lambda-layer-us-west-2-python312-wip:3`

The published versions are made public (`lambda:GetLayerVersion` to `*`), so you
attach them by ARN with no per-account permission grants. Need a different region or
Python version? Let us know and we'll publish it.

!!! note "The `-wip` suffix"
    The layer name carries `-wip` while its contents are still settling, and the
    version (`:1`) bumps on each re-publish. Pin the exact version your function uses.

## Using it from a Lambda

In the Lambda console, **Add a layer → Specify an ARN** and paste the ARN above.
Then import and use `PipelineManager` directly:

```python
from workbench.lambda_layer.pipeline_manager import PipelineManager


def lambda_handler(event, context):
    pm = PipelineManager(f"s3://{event['bucket']}/ml_pipelines/")
    for item in pm.plan():            # (job, run, reason) per job, real AWS mtimes
        if item.run:
            ...  # submit item.job
```

`PipelineManager` uses the default boto3 session (the Lambda's role and region), so
no Workbench config is required.

<img alt="add lambda layer" src="https://github.com/user-attachments/assets/7d0e2fbe-b907-42bc-96bd-3b274d94c3de">

### IAM

The execution role needs read access to whatever artifacts the manager resolves
modification times against (plus the bucket it discovers `pipelines.json` from):

- `glue:GetTable` — DataSource update times
- `sagemaker:DescribeFeatureGroup`, `sagemaker:ListModelPackages` — FeatureSet/Model times
- `s3:GetObject`, `s3:ListBucket` — discovering `pipelines.json` files

See [Workbench Access Controls](https://docs.google.com/presentation/d/1_KwbaBsyBoiWW_8SEallHg8RMsi9FdK10dr2wwzo3CA/edit?usp=sharing).

### Runnable example

A read-only example handler plus an offline smoke test (loads the *built* layer and
runs without AWS) lives in
[`lambda_layers/example_lambda/`](https://github.com/SuperCowPowers/workbench/tree/main/lambda_layers/example_lambda).
Use it to validate the layer in your account before wiring up a real job.

## Exception log forwarding

When a Lambda crashes, the AWS console shows only the last line of the exception.
Wrap your handler to forward the full stack to CloudWatch:

```python
from workbench.utils.workbench_logging import exception_log_forward

with exception_log_forward():
    ...  # your lambda code; any exception/stack is forwarded to CloudWatch
```

## Building and publishing

The layer is built and published from
[`lambda_layers/`](https://github.com/SuperCowPowers/workbench/tree/main/lambda_layers):

```bash
./lambda_layers/build_deploy.sh                                    # build the zip locally
AWS_PROFILE=<profile> ./lambda_layers/build_deploy.sh --deploy     # publish to us-east-1, us-west-2
```

The dependency budget (source + `networkx` + `pandas`) is enforced by
`tests/lambda_layer/test_layer_dependencies.py`, which fails if any `lambda_layer`
module imports outside it.

## Additional Resources

- Setting up Workbench on your AWS Account: [AWS Setup](../aws_setup/core_stack.md)
- Using Workbench for ML Pipelines: [ML Pipelines](../ml_pipelines/index.md)
- Workbench Access Management: [Access Management](https://docs.google.com/presentation/d/1_KwbaBsyBoiWW_8SEallHg8RMsi9FdK10dr2wwzo3CA/edit?usp=sharing)

<img align="right" src="../images/scp.png" width="180">

- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
