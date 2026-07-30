# Workbench Lambda Layers

Build context for the **workbench Lambda layer**: a dependency-minimal slice of
workbench that lambdas can import without bundling the full (heavy) workbench
install.

The layer ships **all of workbench's source** plus only the allowlisted
third-party deps in [`requirements.txt`](requirements.txt) (currently `networkx`
and `pandas`). Other heavy dependencies are intentionally absent — `workbench`'s
top-level import is import-light, and only the
[`workbench.lambda_layer`](../src/workbench/lambda_layer/) subpackage is meant to
be imported from a lambda. That contract is enforced by
[`tests/lambda_layer/test_layer_dependencies.py`](../tests/lambda_layer/test_layer_dependencies.py),
which fails if any `lambda_layer` module imports outside the budget.

`boto3`/`botocore` come from the Lambda runtime and are **not** bundled.

## Build / publish

```bash
./build_deploy.sh                      # build the layer zip locally
AWS_PROFILE=<profile> ./build_deploy.sh --deploy   # publish to us-east-1, us-west-2
```

The bundled workbench is pinned by `WORKBENCH_VERSION` in `build_deploy.sh` and
installed from PyPI, so a build is reproducible from its inputs. Keep that pin in
lockstep with the ML pipeline image's `ARG WORKBENCH_VERSION`; `--workbench-version`
overrides it for a one-off build.

Published layer versions are made public (`lambda:GetLayerVersion` to `*`), so
client accounts attach them by ARN with no per-account permission grants. After
`--deploy`, copy the printed ARNs into [`docs/lambda_layer/index.md`](../docs/lambda_layer/index.md).

To test a layer-carried change before it's released, `--local` builds from the
working tree instead. Those publish to a separate `-dev` layer name and are **not**
made public, so unreleased source never lands on the shared layer:

```bash
AWS_PROFILE=<profile> ./build_deploy.sh --local --deploy
```

## Using it from a lambda

Attach the layer ARN for your region/Python version, then:

```python
from workbench.lambda_layer.pipeline_manager import PipelineManager
```

See [`example_lambda/`](example_lambda/) for a runnable, read-only example handler
plus an offline smoke test that loads the built layer.

To add more of workbench to the layer, move (or add) the module under
`workbench.lambda_layer` and keep it within the dependency budget — if it needs
a new third-party package, add that package to both `requirements.txt` and the
guard's `ALLOWED`.
