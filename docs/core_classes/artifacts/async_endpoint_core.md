# AsyncEndpointCore

!!! tip inline end "API Pass-Through"
    `Endpoint` automatically routes to `AsyncEndpointCore` when the underlying SageMaker endpoint was deployed as async (`workbench_meta["async_endpoint"]`). Callers use [Endpoint](../../api_classes/endpoint.md) — the async S3 round-trip is handled internally.

`AsyncEndpointCore` is the implementation that backs async (long-running) inference for endpoints whose model takes longer than the 60-second sync invocation cap. It supports invocations up to **60 minutes** and scales to zero when idle, so you only pay for compute during active batch runs.

<figure style="margin: 20px auto; text-align: center;">
<img src="../../../images/workbench_async_stack_flow.svg" alt="Async endpoint flow: S3 Upload → SageMaker → Uvicorn → FastAPI → Model → S3 Result" style="width: 100%; min-height: 400px;">
<figcaption><em>Async endpoints add an S3 I/O layer for long-running invocations and scale to zero when idle.</em></figcaption>
</figure>

::: workbench.core.artifacts.async_endpoint_core

## Examples

The examples below use the [Endpoint](../../api_classes/endpoint.md) API class — the same interface you use for sync endpoints. Routing to `AsyncEndpointCore` happens automatically based on the endpoint's deploy-time metadata.

**Run Inference on an Async Endpoint**

```py title="async_endpoint_inference.py"
from workbench.api import Endpoint

# Endpoint detects async deployment and routes through AsyncEndpointCore internally
endpoint = Endpoint("smiles-to-3d-full-v1")
results_df = endpoint.inference(df)
```

**Use with InferenceCache for Batch Processing**

```py title="async_cached_inference.py"
from workbench.api import Endpoint
from workbench.api.inference_cache import InferenceCache

endpoint = Endpoint("smiles-to-3d-full-v1")
cached_endpoint = InferenceCache(endpoint, cache_key_column="smiles")

# Only uncached rows are sent to the endpoint
results_df = cached_endpoint.inference(big_df)
```

**Deploy an Async Endpoint from a Model**

```py title="deploy_async_endpoint.py"
from workbench.api import Model

model = Model("smiles-to-3d-full-v1")
end = model.to_endpoint(
    async_endpoint=True,
    tags=["smiles", "3d descriptors", "full"],
)
# Override the default ml.c7i.xlarge with instance="ml.c7i.2xlarge" if your
# model needs more CPU/memory per worker.
```

Async endpoints deploy with **scale-to-zero** auto-scaling — the instance spins down after ~10 minutes of idle time and cold-starts on the next request. This makes them cost-effective for overnight batch workloads.

## When to Use Async vs Sync

|  | Sync Endpoint | Async Endpoint |
|---|---|---|
| **Invocation timeout** | 60 seconds | 60 minutes |
| **Scaling** | Fixed instance count | Scale-to-zero when idle |
| **Best for** | Realtime inference, low latency | Long-running batch processing |
| **Cost when idle** | Pays for running instance | Zero (scales down) |
| **Caller code** | `Endpoint(name).inference(df)` | `Endpoint(name).inference(df)` (identical) |

The sync/async choice is made at **deploy time** via `model.to_endpoint(async_endpoint=True)`. Caller code is identical in both cases.
