# Graph Store

!!! tip inline end "Graph Storage"
    `GraphStore` provides named, persistent storage for [NetworkX](https://networkx.org/) graphs, backed by AWS S3.

Just like the [DataFrame Store](df_store.md) handles named DataFrames, the `GraphStore` class lets your team list, add, retrieve, and delete named NetworkX graphs from a shared, inspectable location.

!!!tip "Workbench REPL"
    Experiment with listing, adding, and getting graphs using the `GraphStore()` class in the [Workbench REPL](../repl/index.md).

::: workbench.api.graph_store
