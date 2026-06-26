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

    def test_inference(self, *args, **kwargs) -> pd.DataFrame:
        """Smoke-test the Endpoint by running inference on a sample of rows.

        Async endpoints default to a smaller sample (per-row cost is high).

        Returns:
            pd.DataFrame: The DataFrame with predictions
        """
        if self._async is not None:
            return self._async.test_inference(*args, **kwargs)
        return super().test_inference(*args, **kwargs)

    def purge_async_queue(self) -> int:
        """Cancel queued async invocations by deleting their staged S3 inputs.

        Useful when a long-running client was killed and you want to abandon
        the orphaned backlog instead of waiting for the fleet to drain it.
        Only valid on async endpoints — raises on sync endpoints.

        Returns:
            int: Number of staged input objects deleted.
        """
        if self._async is None:
            raise RuntimeError(
                f"Endpoint '{self.name}' is not async — purge_async_queue is only "
                f"meaningful for endpoints with an async invocation queue."
            )
        return self._async.purge_async_queue()

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

    def output_columns(self) -> List[str]:
        """Return this endpoint's registered output columns.

        Works for any endpoint that emits new columns during inference:
        feature endpoints emit computed feature columns; predictor endpoints
        emit prediction / confidence / quantile columns.

        Cached at ``/workbench/endpoints/<name>/output_columns``; lazily
        populated by :func:`workbench.utils.endpoint_utils.register_output_columns`
        on first call (smoke inference) and refreshed when the endpoint is
        redeployed.
        """
        from workbench.utils.endpoint_utils import (
            lookup_cached_columns,
            output_columns_key,
            register_output_columns,
        )

        return lookup_cached_columns(self, output_columns_key(self.name), register_output_columns, "output columns")

    def input_columns(self) -> List[str]:
        """Return this endpoint's declared input columns.

        The columns the endpoint consumes during inference (e.g. ``["smiles"]``
        for a feature endpoint, or the model's training features for a
        predictor endpoint).

        Cached at ``/workbench/endpoints/<name>/input_columns``; lazily
        populated by :func:`workbench.utils.endpoint_utils.register_input_columns`
        on first call (reads ``model.features()``) and refreshed when the
        endpoint is redeployed.
        """
        from workbench.utils.endpoint_utils import (
            input_columns_key,
            lookup_cached_columns,
            register_input_columns,
        )

        return lookup_cached_columns(self, input_columns_key(self.name), register_input_columns, "input columns")

    def inference_batch_size(self) -> int:
        """Return the per-invocation batch size declared for this endpoint.

        Reads ``workbench_meta["inference_batch_size"]`` if set; otherwise
        returns the framework default — 10 for async endpoints, 100 for sync.
        """
        meta = self.workbench_meta() or {}
        if "inference_batch_size" in meta:
            return int(meta["inference_batch_size"])
        return 10 if meta.get("async_endpoint") else 100

    def instance_counts(self) -> dict:
        """Return this endpoint's current and desired instance counts.

        Refreshes endpoint metadata from AWS first so the result reflects
        live autoscaling state, not the snapshot from construction time.

        Shape depends on the endpoint:

        - Regular endpoint: ``{"current": N, "desired": M}``.
        - MetaEndpoint: a flat dict keyed by endpoint name listing the
          meta itself plus each async child, e.g.
          ``{"meta-name":  {"current": 1, "desired": 1},
              "child-a":   {"current": 8, "desired": 8}}``.

        Returns ``{}`` for serverless endpoints (no meaningful instance
        count) or if the AWS describe-endpoint metadata is unavailable.
        """
        self.refresh_meta()
        own = self._read_instance_counts()

        meta = self.workbench_meta() or {}
        dag_dict = meta.get("meta_endpoint_dag")
        if not dag_dict:
            return own

        # Meta endpoint — list itself + async children. Children are read
        # via construction (fresh describe_endpoint) + cached helper to
        # avoid the double-fetch of a recursive instance_counts() call.
        result = {self.name: own}
        async_children = [name for name, is_async in dag_dict.get("endpoint_async", {}).items() if is_async]
        for child_name in async_children:
            result[child_name] = Endpoint(child_name)._read_instance_counts()
        return result


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
    df = my_features.pull_dataframe()
    results = my_endpoint.inference(df)
    target = model.target()
    pprint(results[[target, "prediction"]])

    # Run predictions using the test_inference method
    test_results = my_endpoint.test_inference()
    pprint(test_results[[target, "prediction"]])

    # Run predictions using the fast_inference method
    fast_results = my_endpoint.fast_inference(df)
