# Workbench Sandbox AWS Deployment

To run/deploy the Workbench AWS Sandbox you'll need install a couple of Python packages
either in a Python VirtualENV of your choice (PyENV is good) or any Python3 will do. You'll also need to use a newer version of node such as node v19.6

```
pip install -r requirements.txt
```

At this point you can now synthesize the CloudFormation template and deploy the Workbench Sandbox AWS Components.

```
cdk synth
cdk diff
cdk deploy
```

## Useful commands

 * `cdk ls`          list all stacks in the app
 * `cdk synth`       emits the synthesized CloudFormation template
 * `cdk deploy`      deploy this stack to your default AWS account/region
 * `cdk diff`        compare deployed stack with current state
 * `cdk docs`        open CDK documentation

## CDK Notes
The `cdk.json` file tells the CDK Toolkit how to execute your app.

To add additional dependencies, for example other CDK libraries, just add
them to your `setup.py` file and rerun the `pip install -r requirements.txt`
command.

## Async Endpoint Auto-Scaling

Async SageMaker endpoints deployed via `ModelToEndpoint(..., async_endpoint=True)`
are registered with two Application Auto Scaling policies:

| Policy | Metric | Transition | Why |
| --- | --- | --- | --- |
| **StepScaling** | `HasBacklogWithoutCapacity` (binary 0/1) | 0 → 1 instance | Target-tracking cannot scale from zero — `ApproximateBacklogSizePerInstance` is undefined when instance count is 0 (divide-by-zero). |
| **TargetTracking** | `ApproximateBacklogSizePerInstance` (target 2.0) | 1 → N, N → 0 | Drives steady-state scaling once at least one instance exists. |

Registration happens in [`register_autoscaling()`](../../src/workbench/utils/endpoint_autoscaling.py)
after the endpoint reaches `InService`.

### IAM scope

`endpoint_autoscaling()` in `workbench_core_stack.py` defines four scoped statements:

1. `application-autoscaling:*` on `scalable-target/*` (account/region)
2. `cloudwatch:Put/Delete/DescribeAlarms` on `alarm:TargetTracking-endpoint/*`
3. `sagemaker:UpdateEndpointWeightsAndCapacities` on `endpoint/*`
4. `iam:CreateServiceLinkedRole` conditioned on `iam:AWSServiceName = application-autoscaling.amazonaws.com` (one-time SLR)

### Debugging "stuck at 1 instance"

If an async endpoint isn't scaling out:

1. **Is backlog actually building?** Run `scripts/admin/verify_async_autoscaling.py <name> <sample.csv> --n 32`
   to fire 32 parallel invocations and watch the `HasBacklogWithoutCapacity` / `ApproximateBacklogSizePerInstance`
   timeline. If backlog stays at 0, the caller is submitting sequentially — check
   `AsyncEndpointCore._async_batch_invoke` (should use ThreadPoolExecutor).
2. **Are both policies registered?**
   ```
   aws application-autoscaling describe-scaling-policies \
       --service-namespace sagemaker \
       --resource-id endpoint/<name>/variant/AllTraffic
   ```
   Expect to see *two* policies (step + target) for scale-to-zero endpoints.
3. **Is the step-scaling alarm wired?**
   ```
   aws cloudwatch describe-alarms \
       --alarm-name-prefix TargetTracking-endpoint/<name>/variant/AllTraffic
   ```
   Expect one `-has-backlog` alarm (for 0→1) plus auto-created target-tracking alarms.
4. **Tuning** — override per-model via `workbench_meta`:
   - `async_max_concurrent_per_instance` (default 2) — SageMaker `MaxConcurrentInvocationsPerInstance`.
   - `inference_max_in_flight` (default 16) — client-side parallel submissions.
   - `inference_batch_size` (default 50) — rows per invocation.

