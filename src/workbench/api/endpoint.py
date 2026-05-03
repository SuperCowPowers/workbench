"""Endpoint: Manages AWS Endpoint creation and deployment.
Endpoints are automatically set up and provisioned for deployment into AWS.
Endpoints can be viewed in the AWS Sagemaker interfaces or in the Workbench
Dashboard UI, which provides additional model details and performance metrics"""

from typing import List

import pandas as pd

# Workbench Imports
from workbench.core.artifacts.endpoint_core import EndpointCore


class Endpoint(EndpointCore):
    """Endpoint: Workbench Endpoint API Class

    Common Usage:
        ```python
        my_endpoint = Endpoint(name)
        my_endpoint.details()
        my_endpoint.inference(eval_df)
        ```

    If the underlying endpoint was deployed as async (``workbench_meta["async_endpoint"]``),
    ``inference()`` / ``fast_inference()`` transparently route through an internal
    async core so callers get correct behavior from a single object.

    For feature endpoints (those that emit registered feature columns), use
    :meth:`feature_list` to retrieve the column list.
    """

    def __init__(self, endpoint_name: str):
        super().__init__(endpoint_name)
        self._async = None
        if self.exists() and (self.workbench_meta() or {}).get("async_endpoint"):
            from workbench.core.artifacts.async_endpoint_core import AsyncEndpointCore

            self._async = AsyncEndpointCore(endpoint_name)

    def details(self, **kwargs) -> dict:
        """Endpoint Details

        Returns:
            dict: A dictionary of details about the Endpoint
        """
        return super().details(**kwargs)

    def inference(
        self,
        eval_df: pd.DataFrame,
        capture_name: str = None,
        id_column: str = None,
        drop_error_rows: bool = False,
        include_quantiles: bool = False,
    ) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            capture_name (str, optional): The Name of the capture to use (default: None)
            id_column (str, optional): The name of the column to use as the ID (default: None)
            drop_error_rows (bool): Whether to drop rows with errors (default: False)
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        if self._async is not None:
            return self._async.inference(eval_df, capture_name, id_column, drop_error_rows, include_quantiles)
        return super().inference(eval_df, capture_name, id_column, drop_error_rows, include_quantiles)

    def auto_inference(self) -> pd.DataFrame:
        """Run inference on the Endpoint using the test data from the model training view

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().auto_inference()

    def full_inference(self) -> pd.DataFrame:
        """Run inference on the Endpoint using the full data from the model training view

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        return super().full_inference()

    def ts_inference(self, date_column: str, after_date: str, exclude_ids: list = None) -> pd.DataFrame:
        """Run temporal hold-out inference on this Endpoint.

        Re-runs the temporal split on the FeatureSet data to identify holdout rows
        (those with date > after_date), then runs inference on that holdout set.

        Args:
            date_column (str): Name of the date column.
            after_date (str): Run inference on rows strictly after this date.
            exclude_ids (list): IDs to exclude from the holdout set (e.g., anomalous
                compounds from compute_sample_weights).

        Returns:
            pd.DataFrame: DataFrame with the inference results (empty if no hold-out rows)
        """
        return super().ts_inference(date_column, after_date=after_date, exclude_ids=exclude_ids)

    def fast_inference(self, eval_df: pd.DataFrame, threads: int = 4) -> pd.DataFrame:
        """Run inference on the Endpoint using the provided DataFrame

        Args:
            eval_df (pd.DataFrame): The DataFrame to run predictions on
            threads (int): The number of threads to use (default: 4)

        Returns:
            pd.DataFrame: The DataFrame with predictions

        Note:
            There's no sanity checks or error handling... just FAST Inference!
        """
        if self._async is not None:
            return self._async.fast_inference(eval_df, threads=threads)
        return super().fast_inference(eval_df, threads=threads)

    def cross_fold_inference(self, include_quantiles: bool = False) -> pd.DataFrame:
        """Pull cross-fold inference from model associated with this Endpoint

        Args:
            include_quantiles (bool): Include q_* quantile columns in saved output (default: False)

        Returns:
            pd.DataFrame: A DataFrame with cross fold predictions
        """
        return super().cross_fold_inference(include_quantiles)

    def feature_list(self) -> List[str]:
        """Return this endpoint's registered feature columns.

        Only meaningful for *feature endpoints* — endpoints that take an input
        column (typically ``smiles``) and emit computed feature columns
        (descriptors, fingerprints, etc.).

        Fast path: reads ``/workbench/feature_lists/<endpoint_name>`` from
        ParameterStore (populated by
        :func:`workbench.utils.feature_endpoint_utils.register_features` at
        deploy time).

        Freshness check: compares the parameter's ``LastModifiedDate`` to the
        endpoint's ``modified()`` time. If the endpoint has been redeployed
        since the feature list was cached, the cache is stale — we re-derive
        via the fallback path below and rewrite the cache.

        Fallback (also used when there's no cache yet): runs a small smoke
        inference to discover the columns, writes them to ParameterStore so
        subsequent calls are fast, and returns the list.

        Returns:
            List of feature column names.

        Raises:
            RuntimeError: If the fallback inference fails (e.g. the endpoint
                isn't actually a feature endpoint and produces no new columns).
        """
        from workbench.api.parameter_store import ParameterStore
        from workbench.utils.feature_endpoint_utils import feature_list_key, register_features

        ps = ParameterStore()
        key = feature_list_key(self.name)
        cols = ps.get(key)

        if cols is None:
            self.log.important(
                f"Endpoint[{self.name}]: no feature list registered yet — "
                f"running smoke inference to discover and register columns."
            )
            return register_features(self)

        param_modified = ps.last_modified(key)
        try:
            endpoint_modified = self.modified()
        except Exception:
            endpoint_modified = None

        if param_modified is not None and endpoint_modified is not None and endpoint_modified > param_modified:
            self.log.important(
                f"Endpoint[{self.name}]: endpoint modified at {endpoint_modified} "
                f"is newer than cached feature list ({param_modified}) — re-deriving."
            )
            return register_features(self)

        return cols


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from workbench.api.model import Model
    from workbench.api.feature_set import FeatureSet
    from pprint import pprint

    # Retrieve an existing Data Source
    my_endpoint = Endpoint("abalone-regression")
    pprint(my_endpoint.summary())

    # Run predictions on the Endpoint
    model = Model(my_endpoint.get_input())
    my_features = FeatureSet(model.get_input())
    table = my_features.view("training").table
    df = my_features.query(f'SELECT * FROM "{table}" where training = FALSE')
    results = my_endpoint.inference(df)
    target = model.target()
    pprint(results[[target, "prediction"]])

    # Run predictions using the auto_inference method
    auto_results = my_endpoint.auto_inference()
    pprint(auto_results[[target, "prediction"]])

    # Run predictions using the fast_inference method
    fast_results = my_endpoint.fast_inference(df)
