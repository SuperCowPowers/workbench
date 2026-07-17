# Reports

!!! tip inline end "Published Reports"
    `Reports` is a thin wrapper over the [DataFrame Store](df_store.md) that scopes every operation to the `/reports` subtree.

Writers publish analysis reports (for example, the promotion arbiter publishing contest results) and readers (dashboards, scripts) list and retrieve them. Reads are uncached, so every `get()` reflects the latest published report.

!!!tip "Workbench REPL"
    Experiment with the `Reports()` class in the [Workbench REPL](../repl/index.md).

::: workbench.api.reports
