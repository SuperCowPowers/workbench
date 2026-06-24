# Async Endpoints

!!! info inline end "Same Class, Different Mode"
    There is no separate `AsyncEndpoint` API class. Async behavior lives in [AsyncEndpointCore](../core_classes/artifacts/async_endpoint_core.md) and is invoked transparently by [Endpoint](endpoint.md) when the underlying SageMaker endpoint was deployed as async.

Async endpoints support long-running inference (up to 60 minutes per invocation) and scale to zero when idle, so you only pay for compute during active batch runs. The API is the same as a sync `Endpoint`: **send a DataFrame, get a DataFrame back** — the S3 round-trip is handled internally.

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/workbench_async_stack_flow.svg" alt="Async endpoint flow: S3 Upload → SageMaker → Uvicorn → FastAPI → Model → S3 Result" style="width: 100%; min-height: 400px;">
<figcaption><em>Async endpoints add an S3 I/O layer for long-running invocations and scale to zero when idle.</em></figcaption>
</figure>

## Quick Example

```py title="async_inference.py"
from workbench.api import Endpoint

# Endpoint auto-detects the async deployment and routes accordingly
endpoint = Endpoint("smiles-to-3d-full-v1")
results_df = endpoint.inference(df)
```

## Deploy a New Async Endpoint

```py title="deploy_async.py"
from workbench.api import Model

model = Model("smiles-to-3d-full-v1")
model.to_endpoint(async_endpoint=True, tags=["smiles", "3d descriptors", "full"])
```

## Async Meta Endpoints

An async endpoint can itself be a *composition* of other endpoints. **[`smiles-to-2d-3d-v1`](../models/meta_endpoints.md#featured-smiles-to-2d-3d-v1-2d-3d-in-one-call)** is a [MetaEndpoint](../models/meta_endpoints.md) that fans out to the sync 2D descriptor endpoint and the async 3D descriptor endpoint, then concatenates the results into one ~387-feature DataFrame:

```py title="meta_async.py"
from workbench.api import MetaEndpoint

# Sync 2D + async 3D, composed into a single async endpoint
end = MetaEndpoint("smiles-to-2d-3d-v1")
results_df = end.inference(df)   # input cols + ~313 2D + 74 3D features
```

Because one child (`smiles-to-3d-full-v1`) is async, the whole MetaEndpoint is **auto-deployed async** — the caller sees a single endpoint and a single `inference()` call, with the S3 round-trip and per-child sync/async dispatch handled server-side.

## Full Reference

For the full method list, deployment options, scaling configuration, and advanced usage, see **[AsyncEndpointCore](../core_classes/artifacts/async_endpoint_core.md)**.
