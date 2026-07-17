# PipelineMeta

!!! tip inline end "Pipeline Metadata"
    `PipelineMeta` resolves pipeline configuration from the `PIPELINE_META` environment variable.

`PipelineMeta` reads a JSON pipeline configuration from the `PIPELINE_META` environment variable and exposes it as attributes (`model_name`, `endpoint_name`, `mode`, `serverless`, etc.). It fails hard — raising `RuntimeError` if `PIPELINE_META` is unset or invalid — so pipeline steps don't run with silently missing config.

::: workbench.core.pipelines.pipeline_meta

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
