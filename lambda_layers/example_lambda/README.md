# Example Lambda — workbench layer test harness

A **read-only** stand-in for the client's nightly DT lambda, used to validate the
workbench Lambda layer in the real Lambda runtime *before* migrating the client.
It builds a `PipelineManager` from an ml_pipelines prefix, computes `pm.plan()`,
and returns the plan as JSON. It does **not** submit to SQS.

## 1. Offline smoke test (no AWS)

After building the layer (`lambda_layers/build_deploy.sh`):

```bash
python lambda_layers/example_lambda/local_test.py
```

Loads `workbench`/`networkx` from the *built layer* (not the dev install) and runs
the handler against `sample_pipelines/` with simulated mtimes. Proves the layer is
import-self-sufficient and the plan logic runs end-to-end.

## 2. Deploy to AWS (you run these — mutations)

Publish the layer, then create the function with the layer attached:

```bash
# Publish the layer (notes the version N) -- see lambda_layers/build_deploy.sh
AWS_PROFILE=<scp> ./lambda_layers/build_deploy.sh --deploy

# Package + create the function (python3.12), attaching the layer ARN from above.
# Currently published (bump the trailing version after each re-publish):
#   arn:aws:lambda:us-east-1:507740646243:layer:workbench-lambda-layer-us-east-1-python312-wip:2
#   arn:aws:lambda:us-west-2:507740646243:layer:workbench-lambda-layer-us-west-2-python312-wip:2
cd lambda_layers/example_lambda
zip handler.zip handler.py
aws lambda create-function \
  --function-name workbench-layer-example \
  --runtime python3.12 --handler handler.lambda_handler \
  --role <execution-role-arn> \
  --timeout 60 --memory-size 512 \
  --layers arn:aws:lambda:us-east-1:507740646243:layer:workbench-lambda-layer-us-east-1-python312-wip:2 \
  --zip-file fileb://handler.zip
```

The execution role needs read-only access to the artifacts `PipelineManager`
resolves mtimes against:

- `glue:GetTable` (DataSource update times)
- `sagemaker:DescribeFeatureGroup`, `sagemaker:ListModelPackages` (fs/model times)
- `s3:GetObject`, `s3:ListBucket` on the ml_pipelines bucket (discovery)

## 3. Invoke

Real AWS mtimes (the actual code path the client lambda will run):

```bash
aws lambda invoke --function-name workbench-layer-example \
  --payload '{"pipelines_path": "s3://<bucket>/ml_pipelines/"}' out.json
cat out.json   # body.workbench_version, will_run[], suspect_environment, full plan
```

Offline forward-flood sim in the real runtime (no Glue/SageMaker calls):

```bash
aws lambda invoke --function-name workbench-layer-example \
  --payload '{"pipelines_path": "s3://<bucket>/ml_pipelines/", "simulate_modified": ["ds:foo"]}' out.json
```

Check CloudWatch for the `workbench <version> -- planning from ...` banner and the
`N/M jobs would run` line. A `suspect_environment: true` in the body means nearly
every artifact came back missing — wrong account/region.

## Cleanup

```bash
aws lambda delete-function --function-name workbench-layer-example
```
