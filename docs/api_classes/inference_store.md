# Inference Store

!!! tip inline end "Inference Storage"
    `InferenceStore` manages captured inference results using AWS S3/Parquet/Snappy with Athena queries on top.

The `InferenceStore` class provides a queryable home for inference results across models. Use it to list the models that have captured inference, count rows, and run Athena queries against the stored predictions.

!!!tip "Workbench REPL"
    Experiment with the `InferenceStore()` class in the [Workbench REPL](../repl/index.md).

::: workbench.api.inference_store
