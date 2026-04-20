"""AsyncEndpoint: Workbench API wrapper for async / batch SageMaker endpoints.

Conceptually this is a **batch** endpoint — it's the right tool when:

  * Per-row inference is slow (seconds to minutes) so the realtime 60-second
    HTTP timeout would fail.
  * You have a finite pile of work (e.g. 4,139 compounds) that you want
    chewed through in parallel, then the endpoint goes cold.
  * Idle cost should be zero between jobs — instances scale to 0 after the
    queue drains.

The *transport* is SageMaker's async invocation path (input/output via S3,
the caller polls for completion). We wrap that so ``inference()`` still
returns a DataFrame synchronously and feels identical to ``Endpoint``.

Example:
    ```python
    from workbench.api import AsyncEndpoint

    end = AsyncEndpoint("smiles-to-3d-full-v1")
    df_result = end.inference(my_df)   # 4,000 rows → scales 0 → max_instances,
                                       # processes in parallel, scales back to 0.
    ```

Scaling for these endpoints is handled by the ``"batch"`` mode of
:func:`workbench.utils.endpoint_autoscaling.register_autoscaling` — see its
docstring for the step-scaling policy shape. The mode is configured at
deploy time via ``Model.to_endpoint(async_endpoint=True, max_instances=N)``.
"""

import pandas as pd

from workbench.core.artifacts.async_endpoint_core import AsyncEndpointCore


class AsyncEndpoint(AsyncEndpointCore):
    """Workbench AsyncEndpoint — a batch-style endpoint on SageMaker's async path.

    Semantically this is a *batch endpoint*: it's designed to accept a pile of
    work, scale out to run it in parallel, then drop back to zero instances
    when the queue drains. Use this class for long-per-row inference
    (conformer generation, co-folding, whole-protein scoring, etc.) where the
    realtime 60s timeout would fail and idle cost between jobs must be zero.

    Inherits all functionality from ``AsyncEndpointCore``. This thin wrapper
    exists to match the ``Endpoint`` / ``EndpointCore`` pattern used elsewhere
    in the Workbench API layer.
    """

    def inference(
        self,
        eval_df: pd.DataFrame,
        capture_name: str = None,
        id_column: str = None,
        drop_error_rows: bool = False,
        include_quantiles: bool = False,
    ) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame.

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            capture_name (str, optional): Name of the inference capture (default: None)
            id_column (str, optional): Name of the ID column (default: None)
            drop_error_rows (bool): Whether to drop rows with errors (default: False)
            include_quantiles (bool): Include q_* quantile columns (default: False)

        Returns:
            pd.DataFrame: The DataFrame with inference results
        """
        return super().inference(eval_df, capture_name, id_column, drop_error_rows, include_quantiles)

    def fast_inference(self, eval_df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
        """Run inference on the Endpoint (async path, threads ignored).

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            threads (int): Ignored for async endpoints (kept for API compat)

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().fast_inference(eval_df, threads=threads)
