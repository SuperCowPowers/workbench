"""AsyncEndpoint: Workbench API wrapper for async SageMaker endpoints.

Drop-in replacement for :class:`Endpoint` that uses the async invocation
path internally.  The caller-facing API is identical — ``inference()``
returns a DataFrame synchronously.

Example:
    ```python
    from workbench.api import AsyncEndpoint

    end = AsyncEndpoint("smiles-to-3d-boltzmann-v1")
    df_result = end.inference(my_df)
    ```
"""

import pandas as pd

from workbench.core.artifacts.async_endpoint_core import AsyncEndpointCore


class AsyncEndpoint(AsyncEndpointCore):
    """Workbench AsyncEndpoint API class.

    Inherits all functionality from AsyncEndpointCore. This thin wrapper
    exists to match the Endpoint / EndpointCore pattern used elsewhere
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
