# Meta Endpoint

!!! tip inline end "Meta Endpoints"
    A `MetaEndpoint` is an [Endpoint](endpoint.md) backed by a DAG of other endpoints. See [Meta Endpoints](../models/meta_endpoints.md) for the concept overview.

`MetaEndpoint` subclasses [`Endpoint`](endpoint.md), so wrapping an existing deployed meta endpoint by name works identically. Use `create()` to build and deploy a new meta endpoint from a DAG.

::: workbench.api.meta_endpoint
