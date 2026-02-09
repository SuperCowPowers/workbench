# Inside a Workbench AWS Endpoint: A Modern Web Stack for ML Inference
When you deploy a model endpoint on AWS SageMaker, the default architecture gives you a battle-tested but aging web stack. Workbench takes a different approach — building custom container images with a modern ASGI stack that delivers better performance and native async support. In this blog we'll compare the two architectures and explain why the Workbench stack is a better foundation for production ML inference.

<figure style="margin: 20px 0;">
<img src="../../images/endpoint_architecture.png" alt="Components of a Workbench AWS Endpoint" style="width: 100%;">
<figcaption><em>The layered architecture of a Workbench endpoint: from the custom image down to the model artifacts.</em></figcaption>
</figure>

## The Default SageMaker Stack: Nginx + Gunicorn + Flask
When you follow AWS's canonical "bring your own container" pattern for SageMaker endpoints, you get a three-tier web stack that's been the reference architecture for years:

### Nginx (Reverse Proxy)
Nginx sits at the front, listening on port 8080 (SageMaker's required port). It accepts incoming HTTP requests from the SageMaker runtime infrastructure and forwards them to the application server over a Unix socket. It handles buffering, connection management, and returns 404 for anything that isn't `/ping` or `/invocations`.

### Gunicorn (WSGI Application Server)
Gunicorn is a pre-fork worker server that spawns multiple copies of the Flask application — typically one worker per CPU core. Each worker is an independent OS process running the **synchronous WSGI** protocol, meaning it handles exactly **one request at a time**. When a worker is processing an inference request, it's blocked until that request completes.

### Flask (Web Framework)
Flask defines the two required endpoints: `GET /ping` for health checks and `POST /invocations` for inference. It's lightweight and well-understood, but it's a synchronous WSGI framework — no native support for async I/O, streaming responses, or WebSocket connections.

### The Request Flow
```
SageMaker Runtime → Nginx (:8080) → Unix Socket → Gunicorn (sync workers) → Flask → Model
```

Each request passes through three layers of process/socket boundaries before reaching your model code. Gunicorn's sync workers mean that concurrency is limited to the number of worker processes — and each worker loads a full copy of the model into memory.

## What's Wrong with the Default Stack?
The Nginx/Gunicorn/Flask stack works, but it has real limitations for modern ML inference workloads:

**Synchronous-only processing.** WSGI is a synchronous protocol from 2003. Each Gunicorn worker blocks on a single request. If your inference involves any I/O — loading data, calling external services, batching — the worker sits idle waiting instead of handling other requests.

**No native streaming support.** SageMaker now supports response streaming via `InvokeEndpointWithResponseStream`, but WSGI and Flask can't do this natively. You need a fundamentally different server architecture to stream tokens or partial results back to the client.

**Memory-heavy concurrency.** Gunicorn achieves concurrency by forking worker processes. Each process loads the full Python interpreter and model into memory. Want 8 concurrent requests? You need 8 copies of your model in RAM.

**An aging ecosystem.** Flask and Gunicorn are mature and stable, but the Python web ecosystem has moved on. The ASGI standard, async/await, and frameworks like FastAPI represent the modern approach to building high-performance Python web services.

## The Workbench Stack: Uvicorn + FastAPI

Workbench endpoints replace the entire default stack with a modern ASGI architecture built on two components:

### Uvicorn (ASGI Server)
[Uvicorn](https://www.uvicorn.org/) is a high-performance ASGI server built on `uvloop` (a fast, drop-in replacement for Python's `asyncio` event loop) and `httptools` (a fast HTTP parser based on Node.js's http-parser). It handles HTTP connections directly — no Nginx reverse proxy needed.

Key advantages over Gunicorn + Nginx:

- **Async I/O**: A single Uvicorn worker can handle many concurrent connections using Python's `async/await`. While one request waits on model loading or I/O, other requests proceed.
- **Fewer moving parts**: One server process replaces two (Nginx + Gunicorn). Fewer processes means simpler debugging, fewer configuration files, and fewer failure modes.
- **Native HTTP/1.1 and WebSocket support**: ASGI natively supports streaming responses and bidirectional communication — critical for streaming inference results.

### FastAPI (ASGI Web Framework)
[FastAPI](https://fastapi.tiangolo.com/) is a modern Python web framework built on ASGI and Pydantic. It's what makes Workbench endpoints type-safe and production-ready:

- **Pydantic models for request/response validation**: Input data is validated against typed schemas before your model code ever sees it. Bad requests get clear error messages automatically.
- **Dependency injection**: Shared resources (model loading, configuration) are managed cleanly without global state or singletons.
- **Native async/await**: Endpoint handlers can be `async def`, enabling non-blocking I/O throughout the request lifecycle.

### The Workbench Request Flow
```
SageMaker Runtime → Uvicorn (:8080, ASGI) → FastAPI → Model Script → Model
```

Two layers instead of three. Async instead of sync. Typed schemas instead of manual parsing.

## DataFrame In, DataFrame Out
At the heart of every Workbench endpoint is a simple contract: **send a DataFrame, get a DataFrame back**. The model script layer handles the translation between HTTP and pandas, so your inference code works with familiar DataFrames rather than raw bytes or JSON blobs.

The FastAPI `/invocations` handler orchestrates three functions that every model script defines:

```python
@app.post("/invocations")
async def invoke(request: Request):
    body = await request.body()
    data = inference_module.input_fn(body, content_type)      # → DataFrame
    result = inference_module.predict_fn(data, model)         # → DataFrame
    output_data = inference_module.output_fn(result, accept)  # → CSV/JSON
    return Response(content=output_data)
```

- **`input_fn`** parses the raw request body into a DataFrame — supports both CSV and JSON. 
- **`predict_fn`** runs dataframe through the model, and returns a new DataFrame with predictions appended. 
- **`output_fn`** serializes the result back to CSV or JSON for the response.

This pattern means model scripts are simplier and that calling inference on an endpoint is as easy as sending a DataFrame:

```python
# Grab Endpoint
end = Endpoint("my_awesome_endpoint")

# Grab data that I want to predict on
my_data = <my inference data>

# Send DataFrame to endpoint and get predictions back as a new DataFrame
predictions = end.inference(my_data)
```

### Why DataFrames Matter
This isn't just a convenience — it's a design decision that pays off across the entire ML lifecycle:

**Column preservation.** The input DataFrame passes through prediction intact. ID columns, target values, metadata — everything comes back alongside the predictions. No need to rejoin predictions to your original data by row index and hope the alignment is correct.

**Case-insensitive feature matching.** Workbench model scripts use `match_features_case_insensitive()` to handle column name variations. If your FeatureSet has `LogP` but your eval DataFrame has `logp`, inference still works — the model script renames columns to match the model's expectations automatically.

**Type handling across the wire.** CSV serialization strips type information — everything becomes a string. The client-side `_predict` method handles the round-trip automatically: numeric columns are converted back with `pd.to_numeric()`, `N/A` placeholders (used because CSV can't represent `NaN` natively) are restored to proper `NaN` values, `__NA__` placeholders for `pd.NA` survive the round-trip, and boolean strings (`"true"/"false"`) are converted back to Python booleans. The result is a DataFrame that looks like you never serialized it at all.

**Consistent interface across model frameworks.** Whether your model is XGBoost, PyTorch, or ChemProp, the contract is the same: DataFrame in, DataFrame out. The model script handles framework-specific details (loading XGBoost models, running ChemProp graph inference, expanding classifier probability columns) while the caller always works with the same pandas interface.

## Custom Image: More Than Just the Web Stack
The Workbench custom image isn't only about Uvicorn and FastAPI — it's a purpose-built environment for computational chemistry and ADMET modeling. The image comes pre-loaded with:

| Package | Purpose |
|---------|---------|
| **RDKit** | Molecular parsing, descriptor computation, substructure search |
| **Mordred** | Additional molecular descriptors (ADMET-focused modules) |
| **ChemProp** | Message-passing neural network (MPNN) inference for molecular property prediction |
| **XGBoost** | Gradient-boosted tree model inference |
| **PyTorch** | Neural network inference (ChemProp backend) |
| **scikit-learn** | Classical ML model inference and preprocessing |

This means endpoint model scripts can import these packages directly without bundling them into the model artifact. The container image handles the complex dependency chain (RDKit's C++ extensions, PyTorch's CUDA bindings, Mordred's dependency on NetworkX) so your model script stays focused on inference logic.

## Side-by-Side Comparison

| Aspect | Default Stack | Workbench Stack |
|--------|--------------|-----------------|
| **Web Server** | Nginx + Gunicorn | Uvicorn |
| **Framework** | Flask (WSGI) | FastAPI (ASGI) |
| **Protocol** | WSGI (synchronous) | ASGI (async-native) |
| **Concurrency Model** | Process forking (1 req/worker) | Async event loop (many req/worker) |
| **Streaming** | Not natively supported | Native ASGI streaming |
| **Request Validation** | Manual | Automatic (Pydantic) |
| **Process Count** | 3 (Nginx + Gunicorn + Flask) | 1 (Uvicorn + FastAPI) |
| **Chemistry Packages** | Install yourself | Pre-loaded (RDKit, Mordred, ChemProp) |

## Why It Matters for ML Inference
The architecture differences aren't academic — they translate directly to operational benefits:

**Simpler debugging.** One server process with structured FastAPI logging beats digging through Nginx access logs, Gunicorn error logs, and Flask tracebacks across three processes.

**Better resource utilization.** Async I/O means a single worker can overlap model loading, preprocessing, and response serialization. You're not paying for idle workers blocked on I/O.

**Ready for advanced patterns.** Streaming inference results, health checks with detailed model status, batch preprocessing with async gathering — these patterns fall out naturally from the ASGI architecture. With WSGI, each one requires workarounds.

**Production-ready chemistry stack.** The custom image means you don't spend hours debugging RDKit compilation issues or Mordred version conflicts inside a SageMaker container. Deploy your model script, and the chemistry packages are already there.

## Robust Error Handling: Binary Search for Bad Rows
Production inference means dealing with messy data — invalid SMILES strings, malformed feature values, and edge cases that crash model scripts. The default SageMaker pattern gives you nothing here: if a single bad row exists in your batch, the entire request fails with a `ModelError` and you get no predictions back.

Workbench takes a fundamentally different approach. When the endpoint returns a `ModelError`, Workbench automatically **bisects** the batch and retries each half recursively. This binary search narrows down to the exact problematic row(s) while still returning predictions for every valid row:

```python
def _endpoint_error_handling(self, predictor, feature_df, drop_error_rows=False):
    try:
        results = predictor.predict(csv_buffer.getvalue())
        return pd.DataFrame.from_records(results[1:], columns=results[0])

    except botocore.exceptions.ClientError as err:
        if err.response["Error"]["Code"] == "ModelError":
            # Base case: single row that fails
            if len(feature_df) == 1:
                if drop_error_rows:
                    return pd.DataFrame(columns=feature_df.columns)
                return self._fill_with_nans(feature_df)  # NaN placeholder

            # Binary search: split and retry both halves
            mid_point = len(feature_df) // 2
            first_half = self._endpoint_error_handling(predictor, feature_df.iloc[:mid_point])
            second_half = self._endpoint_error_handling(predictor, feature_df.iloc[mid_point:])
            return pd.concat([first_half, second_half], ignore_index=True)
```

The algorithm handles several scenarios gracefully:

- **`ModelNotReadyException`**: Sleeps and retries — common with serverless endpoints that cold-start.
- **`ModelError` with multiple rows**: Bisects the batch recursively until the bad row(s) are isolated.
- **Single bad row**: Either fills with NaN placeholders (preserving row alignment) or drops the row entirely, based on the `drop_error_rows` parameter.
- **Unexpected errors**: Logs full error context and raises — no silent failures.

This means you can send 10,000 rows to an endpoint, have 3 of them contain invalid data, and get back 10,000 predictions — 9,997 real values and 3 NaN placeholders. The alternative? Manually chunking your data, catching errors, and hoping you can figure out which rows caused the failure. Workbench handles all of this automatically with logarithmic overhead (a batch of 1,000 with one bad row requires only ~10 extra endpoint calls to isolate it).

The `N/A` → `NaN` conversion and automatic type recovery in `_predict` further smooths over the rough edges of CSV serialization — numeric columns that SageMaker's CSV deserializer returns as strings get converted back to proper numeric types, and `pd.NA` placeholders survive the round-trip through `__NA__` encoding.

## Summary
AWS SageMaker's default endpoint architecture — Nginx, Gunicorn, and Flask — is a proven stack that's been serving models reliably for years. But it's a synchronous, WSGI-era design that shows its age when you need async processing, or streaming responses.

Workbench replaces this with Uvicorn and FastAPI: a modern ASGI stack that's simpler (fewer processes), more capable (async, streaming, auto-docs), and purpose-built for computational chemistry workloads. The DataFrame-in/DataFrame-out contract means model scripts work with familiar pandas code while the framework handles serialization, type recovery, and case-insensitive feature matching automatically. On top of that, robust error handling uses binary search to isolate bad rows in inference batches — so a single malformed SMILES string doesn't take down your entire prediction run. Combined with a custom image pre-loaded with RDKit, Mordred, and ChemProp, Workbench endpoints give you a production-ready ML inference platform without the infrastructure headaches.

## References

- **FastAPI**: Ramírez, S. *FastAPI: Modern, Fast (high-performance) Web Framework for Building APIs.* [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Uvicorn**: Encode. *Uvicorn: An ASGI Web Server.* [https://www.uvicorn.org/](https://www.uvicorn.org/)
- **ASGI Specification**: Django Software Foundation. *Asynchronous Server Gateway Interface.* [https://asgi.readthedocs.io/](https://asgi.readthedocs.io/)
- **SageMaker Custom Containers**: AWS. *Use Your Own Inference Code with Hosting Services.* [https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)
- **Gunicorn**: Chesneau, B. *Gunicorn: Python WSGI HTTP Server for UNIX.* [https://gunicorn.org/](https://gunicorn.org/)
- **RDKit**: [https://github.com/rdkit/rdkit](https://github.com/rdkit/rdkit)
- **ChemProp**: [https://github.com/chemprop/chemprop](https://github.com/chemprop/chemprop)
- **Mordred**: [https://github.com/mordred-descriptor/mordred](https://github.com/mordred-descriptor/mordred)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
