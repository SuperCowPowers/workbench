"""FeatureSet: Workbench Feature Set accessible through Athena"""

import sys
import time
from datetime import datetime, timezone

import botocore.exceptions
import pandas as pd
import awswrangler as wr
import numpy as np

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_store import FeatureStore

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.data_source_factory import DataSourceFactory
from workbench.core.artifacts.athena_source import AthenaSource
from workbench.utils.deprecated_utils import deprecated

from typing import TYPE_CHECKING, Optional, List, Dict, Union

from workbench.utils.aws_utils import aws_throttle

if TYPE_CHECKING:
    from workbench.core.views import View


class FeatureSetCore(Artifact):
    """FeatureSetCore: Workbench FeatureSetCore Class

    Common Usage:
        ```python
        my_features = FeatureSetCore(feature_name)
        my_features.summary()
        my_features.details()
        ```
    """

    def __init__(self, feature_set_name: str, **kwargs):
        """FeatureSetCore Initialization

        Args:
            feature_set_name (str): Name of Feature Set
        """

        # Make sure the feature_set name is valid
        self.is_name_valid(feature_set_name)

        # Call superclass init
        super().__init__(feature_set_name, **kwargs)

        # Get our FeatureSet metadata
        self.feature_meta = self.meta.feature_set(self.name)

        # Sanity check and then set up our FeatureSet attributes
        if self.feature_meta is None:
            self.log.warning(f"Could not find feature set {self.name} within current visibility scope")
            self.data_source = None
            return
        else:
            self.id_column = self.feature_meta["RecordIdentifierFeatureName"]
            self.event_time = self.feature_meta["EventTimeFeatureName"]

            # Pull Athena and S3 Storage information from metadata
            self.athena_table = self.feature_meta["workbench_meta"]["athena_table"]
            self.athena_database = self.feature_meta["workbench_meta"]["athena_database"]
            self.s3_storage = self.feature_meta["workbench_meta"].get("s3_storage")

            # Create our internal DataSource (hardcoded to Athena for now)
            self.data_source = AthenaSource(self.athena_table, self.athena_database)

            # Check our DataSource (AWS Metadata refresh can fix)
            if not self.data_source.exists():
                self.log.warning(
                    f"FS: Data Source {self.athena_table} not found, sleeping and refreshing AWS Metadata..."
                )
                time.sleep(3)
                self.refresh_meta()

        # Spin up our Feature Store
        self.feature_store = FeatureStore(self.sm_session)

        # Call superclass post_init
        super().__post_init__()

        # All done
        self.log.info(f"FeatureSet Initialized: {self.name}...")

    @property
    def table(self) -> str:
        """Get the base table name for this FeatureSet"""
        return self.data_source.table

    def refresh_meta(self):
        """Internal: Refresh our internal AWS Feature Store metadata"""
        self.log.info(f"Calling refresh_meta() on the FeatureSet {self.name}")
        self.feature_meta = self.meta.feature_set(self.name)
        self.id_column = self.feature_meta["RecordIdentifierFeatureName"]
        self.event_time = self.feature_meta["EventTimeFeatureName"]
        self.athena_table = self.feature_meta["workbench_meta"]["athena_table"]
        self.athena_database = self.feature_meta["workbench_meta"]["athena_database"]
        self.s3_storage = self.feature_meta["workbench_meta"].get("s3_storage")
        self.data_source = AthenaSource(self.athena_table, self.athena_database)
        self.log.info(f"Calling refresh_meta() on the DataSource {self.data_source.name}")
        self.data_source.refresh_meta()

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.feature_meta is None:
            self.log.debug(f"FeatureSet {self.name} not found in AWS Metadata!")
            return False
        return True

    def health_check(self) -> list[str]:
        """Perform a health check on this model

        Returns:
            list[str]: List of health issues
        """
        # Call the base class health check
        health_issues = super().health_check()

        # If we have a 'needs_onboard' in the health check then just return
        if "needs_onboard" in health_issues:
            return health_issues

        # Check our DataSource
        if not self.data_source.exists():
            self.log.critical(f"Data Source check failed for {self.name}")
            self.log.critical("Delete this Feature Set and recreate it to fix this issue")
            health_issues.append("data_source_missing")
        return health_issues

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.feature_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.feature_meta["FeatureGroupArn"]

    def size(self) -> float:
        """Return the size of the internal DataSource in MegaBytes"""
        return self.data_source.size()

    @property
    def columns(self) -> list[str]:
        """Return the column names of the Feature Set"""
        return list(self.column_details().keys())

    @property
    def column_types(self) -> list[str]:
        """Return the column types of the Feature Set"""
        return list(self.column_details().values())

    def column_details(self) -> dict:
        """Return the column details of the Feature Set

        Returns:
            dict: The column details of the Feature Set

        Notes:
            We can't call just call self.data_source.column_details() because FeatureSets have different
            types, so we need to overlay that type information on top of the DataSource type information
        """
        fs_details = {item["FeatureName"]: item["FeatureType"] for item in self.feature_meta["FeatureDefinitions"]}
        ds_details = self.data_source.column_details()

        # Overlay the FeatureSet type information on top of the DataSource type information
        for col, dtype in ds_details.items():
            ds_details[col] = fs_details.get(col, dtype)
        return ds_details

    def views(self) -> list[str]:
        """Return the views for this Data Source"""
        from workbench.core.views.view_utils import list_views

        return list_views(self.data_source)

    def supplemental_data(self) -> list[str]:
        """Return the supplemental data for this Data Source"""
        from workbench.core.views.view_utils import list_supplemental_data

        return list_supplemental_data(self.data_source)

    def view(self, view_name: str) -> "View":
        """Return a DataFrame for a specific view
        Args:
            view_name (str): The name of the view to return
        Returns:
            pd.DataFrame: A DataFrame for the specified view
        """
        from workbench.core.views import View

        return View(self, view_name)

    def set_display_columns(self, display_columns: list[str]):
        """Set the display columns for this Data Source

        Args:
            display_columns (list[str]): The display columns for this Data Source
        """
        # Check mismatch of display columns to computation columns
        c_view = self.view("computation")
        computation_columns = c_view.columns
        mismatch_columns = [col for col in display_columns if col not in computation_columns]
        if mismatch_columns:
            self.log.monitor(f"Display View/Computation mismatch: {mismatch_columns}")

        self.log.important(f"Setting Display Columns...{display_columns}")
        from workbench.core.views import DisplayView

        # Create a NEW display view
        DisplayView.create(self, source_table=c_view.table, column_list=display_columns)

    def set_computation_columns(self, computation_columns: list[str], reset_display: bool = True):
        """Set the computation columns for this FeatureSet

        Args:
            computation_columns (list[str]): The computation columns for this FeatureSet
            reset_display (bool): Also reset the display columns to match (default: True)
        """
        self.log.important(f"Setting Computation Columns...{computation_columns}")
        from workbench.core.views import ComputationView

        # Create a NEW computation view
        ComputationView.create(self, column_list=computation_columns)
        self.recompute_stats()

        # Reset the display columns to match the computation columns
        if reset_display:
            self.set_display_columns(computation_columns)

    def set_compressed_features(self, compressed_columns: list[str]):
        """Set the compressed features for this FeatureSet

        Args:
            compressed_columns (list[str]): The compressed columns for this FeatureSet
        """
        # Ensure that the compressed features are a subset of the columns
        if not set(compressed_columns).issubset(set(self.columns)):
            self.log.warning(
                f"Compressed columns {compressed_columns} are not a subset of the columns {self.columns}. "
            )
            return

        # Set the compressed features in our FeatureSet metadata
        self.log.important(f"Setting Compressed Columns...{compressed_columns}")
        self.upsert_workbench_meta({"compressed_features": compressed_columns})

    def get_compressed_features(self) -> list[str]:
        """Get the compressed features for this FeatureSet

        Returns:
            list[str]: The compressed columns for this FeatureSet
        """
        # Get the compressed features from our FeatureSet metadata
        return self.workbench_meta().get("compressed_features", [])

    def num_columns(self) -> int:
        """Return the number of columns of the Feature Set"""
        return len(self.columns)

    def num_rows(self) -> int:
        """Return the number of rows of the internal DataSource"""
        return self.data_source.num_rows()

    def query(self, query: str, overwrite: bool = True) -> pd.DataFrame:
        """Query the internal DataSource

        Args:
            query (str): The query to run against the DataSource
            overwrite (bool): Overwrite the table name in the query (default: True)

        Returns:
            pd.DataFrame: The results of the query
        """
        if overwrite:
            query = query.replace(" " + self.name + " ", " " + self.athena_table + " ")
        return self.data_source.query(query)

    def aws_url(self):
        """The AWS URL for looking at/querying the underlying data source"""
        workbench_details = self.data_source.workbench_meta().get("workbench_details", {})
        return workbench_details.get("aws_url", "unknown")

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.feature_meta["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        # Note: We can't currently figure out how to this from AWS Metadata
        return self.feature_meta["CreationTime"]

    def hash(self) -> str:
        """Return the hash for the set of Parquet files for this artifact"""
        return self.data_source.hash()

    def table_hash(self) -> str:
        """Return the hash for the Athena table"""
        return self.data_source.table_hash()

    def get_data_source(self) -> DataSourceFactory:
        """Return the underlying DataSource object"""
        return self.data_source

    def get_feature_store(self) -> FeatureStore:
        """Return the underlying AWS FeatureStore object. This can be useful for more advanced usage
        with create_dataset() such as Joins and time ranges and a host of other options
        See: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        """
        return self.feature_store

    def create_s3_training_data(self) -> str:
        """Create some Training Data (S3 CSV) from a Feature Set using standard options. If you want
        additional options/features use the get_feature_store() method and see AWS docs for all
        the details: https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store-create-a-dataset.html
        Returns:
            str: The full path/file for the CSV file created by Feature Store create_dataset()
        """

        # Set up the S3 Query results path
        date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
        s3_output_path = self.feature_sets_s3_path + f"/{self.name}/datasets/all_{date_time}"

        # Make the query
        table = self.view("training").table
        query = f'SELECT * FROM "{table}"'
        athena_query = FeatureGroup(name=self.name, sagemaker_session=self.sm_session).athena_query()
        athena_query.run(query, output_location=s3_output_path)
        athena_query.wait()
        query_execution = athena_query.get_query_execution()

        # Get the full path to the S3 files with the results
        full_s3_path = s3_output_path + f"/{query_execution['QueryExecution']['QueryExecutionId']}.csv"
        return full_s3_path

    def get_training_data(self) -> pd.DataFrame:
        """Get the training data for this FeatureSet

        Returns:
            pd.DataFrame: The training data for this FeatureSet
        """
        from workbench.core.views.view import View

        return View(self, "training").pull_dataframe(limit=1_000_000)

    def snapshot_query(self, table_name: str = None) -> str:
        """An Athena query to get the latest snapshot of features

        Args:
            table_name (str): The name of the table to query (default: None)

        Returns:
            str: The Athena query to get the latest snapshot of features
        """
        # Remove FeatureGroup metadata columns that might have gotten added
        columns = self.columns
        filter_columns = ["write_time", "api_invocation_time", "is_deleted"]
        columns = ", ".join(['"' + x + '"' for x in columns if x not in filter_columns])

        query = (
            f"SELECT {columns} "
            f"    FROM (SELECT *, row_number() OVER (PARTITION BY {self.id_column} "
            f"        ORDER BY {self.event_time} desc, api_invocation_time DESC, write_time DESC) AS row_num "
            f'        FROM "{table_name}") '
            "    WHERE row_num = 1 and NOT is_deleted;"
        )
        return query

    def details(self) -> dict[dict]:
        """Additional Details about this FeatureSet Artifact

        Returns:
            dict(dict): A dictionary of details about this FeatureSet
        """

        self.log.info(f"Computing FeatureSet Details ({self.name})...")
        details = self.summary()
        details["aws_url"] = self.aws_url()

        # Store the AWS URL in the Workbench Metadata
        # FIXME: We need to revisit this but doing an upsert just for aws_url is silly
        # self.upsert_workbench_meta({"aws_url": details["aws_url"]})

        # Now get a summary of the underlying DataSource
        details["storage_summary"] = self.data_source.summary()

        # Number of Columns
        details["num_columns"] = self.num_columns()

        # Number of Rows
        details["num_rows"] = self.num_rows()

        # Additional Details
        details["workbench_status"] = self.get_status()
        details["workbench_input"] = self.get_input()
        details["workbench_tags"] = self.tag_delimiter.join(self.get_tags())

        # Underlying Storage Details
        details["storage_type"] = "athena"  # TODO: Add RDS support
        details["storage_name"] = self.data_source.name

        # Add the column details and column stats
        details["column_details"] = self.column_details()
        details["column_stats"] = self.column_stats()

        # Return the details data
        return details

    def delete(self):
        """Instance Method: Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects"""
        # Make sure the AthenaSource exists
        if not self.exists():
            self.log.warning(f"Trying to delete an FeatureSet that doesn't exist: {self.name}")

        # Call the Class Method to delete the FeatureSet
        FeatureSetCore.managed_delete(self.name)

    @classmethod
    def managed_delete(cls, feature_set_name: str):
        """Class Method: Delete the Feature Set: Feature Group, Catalog Table, and S3 Storage Objects

        Args:
            feature_set_name (str): The Name of the FeatureSet to delete
        """

        # See if the FeatureSet exists
        try:
            response = cls.sm_client.describe_feature_group(FeatureGroupName=feature_set_name)
        except cls.sm_client.exceptions.ResourceNotFound:
            cls.log.info(f"FeatureSet {feature_set_name} not found!")
            return

        # Extract database and table information from the response
        offline_config = response.get("OfflineStoreConfig", {})
        database = offline_config.get("DataCatalogConfig", {}).get("Database")
        offline_table = offline_config.get("DataCatalogConfig", {}).get("TableName")
        data_source_name = offline_table  # Our offline storage IS a DataSource

        # Delete the Feature Group and ensure that it gets deleted
        cls.log.important(f"Deleting FeatureSet {feature_set_name}...")
        remove_fg = cls.aws_feature_group_delete(feature_set_name)
        cls.ensure_feature_group_deleted(remove_fg)

        # Delete our underlying DataSource (Data Catalog Table and S3 Storage Objects)
        AthenaSource.managed_delete(data_source_name, database=database)

        # Delete any views associated with this FeatureSet
        cls.delete_views(offline_table, database)

        # Feature Sets can often have a lot of cruft so delete the entire bucket/prefix
        s3_delete_path = cls.feature_sets_s3_path + f"/{feature_set_name}/"
        cls.log.info(f"Deleting All FeatureSet S3 Storage Objects {s3_delete_path}")
        wr.s3.delete_objects(s3_delete_path, boto3_session=cls.boto3_session)

        # Delete any dataframes that were stored in the Dataframe Cache
        cls.log.info("Deleting Dataframe Cache...")
        cls.df_cache.delete_recursive(feature_set_name)

    @classmethod
    @aws_throttle
    def aws_feature_group_delete(cls, feature_set_name):
        remove_fg = FeatureGroup(name=feature_set_name, sagemaker_session=cls.sm_session)
        remove_fg.delete()
        return remove_fg

    @classmethod
    def ensure_feature_group_deleted(cls, feature_group):
        status = "Deleting"
        while status == "Deleting":
            cls.log.debug("FeatureSet being Deleted...")
            try:
                status = feature_group.describe().get("FeatureGroupStatus")
            except botocore.exceptions.ClientError as error:
                # For ResourceNotFound/ValidationException, this is fine, otherwise raise all other exceptions
                if error.response["Error"]["Code"] in ["ResourceNotFound", "ValidationException"]:
                    break
                else:
                    raise error
            time.sleep(1)
        cls.log.info(f"FeatureSet {feature_group.name} successfully deleted")

    def set_training_holdouts(self, holdout_ids: list[str]):
        """Set the hold out ids for the training view for this FeatureSet

        Args:
            holdout_ids (list[str]): The list of holdout ids.
        """
        from workbench.core.views import TrainingView

        # Create a NEW training view
        self.log.important(f"Setting Training Holdouts: {len(holdout_ids)} ids...")
        TrainingView.create(self, id_column=self.id_column, holdout_ids=holdout_ids)

    def get_training_holdouts(self) -> list[str]:
        """Get the hold out ids for the training view for this FeatureSet

        Returns:
            list[str]: The list of holdout ids.
        """

        # Create a NEW training view
        self.log.important("Getting Training Holdouts...")
        table = self.view("training").table
        hold_out_ids = self.query(f'SELECT {self.id_column} FROM "{table}" where training = FALSE')[
            self.id_column
        ].tolist()
        return hold_out_ids

    def set_sample_weights(
        self,
        weight_dict: Dict[Union[str, int], float],
        default_weight: float = 1.0,
        exclude_zero_weights: bool = True,
    ):
        """Configure training view with sample weights for each ID.

        Args:
            weight_dict: Mapping of ID to sample weight
                - weight > 1.0: oversample/emphasize
                - weight = 1.0: normal (default)
                - 0 < weight < 1.0: downweight/de-emphasize
                - weight = 0.0: exclude from training
            default_weight: Weight for IDs not in weight_dict (default: 1.0)
            exclude_zero_weights: If True, filter out rows with sample_weight=0 (default: True)

        Example:
            weights = {
                'compound_42': 3.0,  # oversample 3x
                'compound_99': 0.1,  # noisy, downweight
                'compound_123': 0.0, # exclude from training
            }
            model.set_sample_weights(weights)  # zeros automatically excluded
            model.set_sample_weights(weights, exclude_zero_weights=False)  # keep zeros
        """
        from workbench.core.views import TrainingView

        if not weight_dict:
            self.log.important("Empty weight_dict, creating standard training view")
            TrainingView.create(self, id_column=self.id_column)
            return

        self.log.important(f"Setting sample weights for {len(weight_dict)} IDs")

        # Helper to format IDs for SQL
        def format_id(id_val):
            return repr(id_val)

        # Build CASE statement for sample_weight
        case_conditions = [
            f"WHEN {self.id_column} = {format_id(id_val)} THEN {weight}" for id_val, weight in weight_dict.items()
        ]
        case_statement = "\n        ".join(case_conditions)

        # Build inner query with sample weights
        inner_sql = f"""SELECT
            *,
            CASE
                {case_statement}
                ELSE {default_weight}
            END AS sample_weight
        FROM {self.table}"""

        # Optionally filter out zero weights
        if exclude_zero_weights:
            zero_count = sum(1 for weight in weight_dict.values() if weight == 0.0)
            custom_sql = f"SELECT * FROM ({inner_sql}) WHERE sample_weight > 0"
            self.log.important(f"Filtering out {zero_count} rows with sample_weight = 0")
        else:
            custom_sql = inner_sql

        TrainingView.create_with_sql(self, sql_query=custom_sql, id_column=self.id_column)

    @deprecated(version="0.9")
    def set_training_filter(self, filter_expression: Optional[str] = None):
        """Set a filter expression for the training view for this FeatureSet

        Args:
            filter_expression (Optional[str]): A SQL filter expression (e.g., "age > 25 AND status = 'active'")
                If None or empty string, will reset to training view with no filter
                (default: None)
        """
        from workbench.core.views import TrainingView

        # Grab the existing holdout ids
        holdout_ids = self.get_training_holdouts()

        # Create a NEW training view
        self.log.important(f"Setting Training Filter: {filter_expression}")
        TrainingView.create(
            self, id_column=self.id_column, holdout_ids=holdout_ids, filter_expression=filter_expression
        )

    @deprecated(version="0.9")
    def exclude_ids_from_training(self, ids: List[Union[str, int]], column_name: Optional[str] = None):
        """Exclude a list of IDs from the training view

        Args:
            ids (List[Union[str, int]],): List of IDs to exclude from training
            column_name (Optional[str]): Column name to filter on.
                If None, uses self.id_column (default: None)
        """
        # Use the default id_column if not specified
        column = column_name or self.id_column

        # Handle empty list case
        if not ids:
            self.log.warning("No IDs provided to exclude")
            return

        # Build the filter expression with proper SQL quoting
        quoted_ids = ", ".join([repr(id) for id in ids])
        filter_expression = f"{column} NOT IN ({quoted_ids})"

        # Apply the filter
        self.set_training_filter(filter_expression)

    @deprecated(version="0.9")
    def set_training_sampling(
        self,
        exclude_ids: Optional[List[Union[str, int]]] = None,
        replicate_ids: Optional[List[Union[str, int]]] = None,
        replication_factor: int = 2,
    ):
        """Configure training view with ID exclusions and replications (oversampling).

        Args:
            exclude_ids: List of IDs to exclude from training view
            replicate_ids: List of IDs to replicate in training view for oversampling
            replication_factor: Number of times to replicate each ID (default: 2)

        Note:
            If an ID appears in both lists, exclusion takes precedence.
        """
        from workbench.core.views import TrainingView

        # Normalize to empty lists if None
        exclude_ids = exclude_ids or []
        replicate_ids = replicate_ids or []

        # Remove any replicate_ids that are also in exclude_ids (exclusion wins)
        replicate_ids = [rid for rid in replicate_ids if rid not in exclude_ids]

        # If no sampling needed, just create normal view
        if not exclude_ids and not replicate_ids:
            self.log.important("No sampling specified, creating standard training view")
            TrainingView.create(self, id_column=self.id_column)
            return

        # Build the custom SQL query
        self.log.important(
            f"Excluding {len(exclude_ids)} IDs, Replicating {len(replicate_ids)} IDs "
            f"(factor: {replication_factor}x)"
        )

        # Helper to format IDs for SQL
        def format_ids(ids):
            return ", ".join([repr(id) for id in ids])

        # Start with base query
        base_query = f"SELECT * FROM {self.table}"

        # Add exclusions if needed
        if exclude_ids:
            base_query += f"\nWHERE {self.id_column} NOT IN ({format_ids(exclude_ids)})"

        # Build full query with replication
        if replicate_ids:
            # Generate VALUES clause for CROSS JOIN: (1), (2), ..., (N-1)
            # We want N-1 additional copies since the original row is already in base_query
            values_clause = ", ".join([f"({i})" for i in range(1, replication_factor)])

            custom_sql = f"""{base_query}

            UNION ALL

            SELECT t.*
            FROM {self.table} t
            CROSS JOIN (VALUES {values_clause}) AS n(num)
            WHERE t.{self.id_column} IN ({format_ids(replicate_ids)})"""
        else:
            # Only exclusions, no UNION needed
            custom_sql = base_query

        # Create the training view with our custom SQL
        TrainingView.create_with_sql(self, sql_query=custom_sql, id_column=self.id_column)

    @classmethod
    def delete_views(cls, table: str, database: str):
        """Delete any views associated with this FeatureSet

        Args:
            table (str): Name of Athena Table
            database (str): Athena Database Name
        """
        from workbench.core.views.view_utils import delete_views_and_supplemental_data

        delete_views_and_supplemental_data(table, database, cls.boto3_session)

    def descriptive_stats(self) -> dict:
        """Get the descriptive stats for the numeric columns of the underlying DataSource

        Returns:
            dict: A dictionary of descriptive stats for the numeric columns
        """
        return self.data_source.descriptive_stats()

    def sample(self) -> pd.DataFrame:
        """Get a sample of the data from the underlying DataSource

        Returns:
            pd.DataFrame: A sample of the data from the underlying DataSource
        """
        return self.data_source.sample()

    def outliers(self, scale: float = 1.5) -> pd.DataFrame:
        """Compute outliers for all the numeric columns in a DataSource

        Args:
            scale (float): The scale to use for the IQR (default: 1.5)
        Returns:
            pd.DataFrame: A DataFrame of outliers from this DataSource
        Notes:
            Uses the IQR * 1.5 (~= 2.5 Sigma) method to compute outliers
            The scale parameter can be adjusted to change the IQR multiplier
        """
        return self.data_source.outliers(scale=scale)

    def smart_sample(self) -> pd.DataFrame:
        """Get a SMART sample dataframe from this FeatureSet

        Returns:
            pd.DataFrame: A combined DataFrame of sample data + outliers
        """
        return self.data_source.smart_sample()

    def anomalies(self) -> pd.DataFrame:
        """Get a set of anomalous data from the underlying DataSource
        Returns:
            pd.DataFrame: A dataframe of anomalies from the underlying DataSource
        """

        # FIXME: Mock this for now
        anom_df = self.sample().copy()
        anom_df["anomaly_score"] = np.random.rand(anom_df.shape[0])
        anom_df["cluster"] = np.random.randint(0, 10, anom_df.shape[0])
        anom_df["x"] = np.random.rand(anom_df.shape[0])
        anom_df["y"] = np.random.rand(anom_df.shape[0])
        return anom_df

    def value_counts(self) -> dict:
        """Get the value counts for the string columns of the underlying DataSource

        Returns:
            dict: A dictionary of value counts for the string columns
        """
        return self.data_source.value_counts()

    def correlations(self) -> dict:
        """Get the correlations for the numeric columns of the underlying DataSource

        Returns:
            dict: A dictionary of correlations for the numeric columns
        """
        return self.data_source.correlations()

    def column_stats(self) -> dict[dict]:
        """Compute Column Stats for all the columns in the FeatureSets underlying DataSource

        Returns:
            dict(dict): A dictionary of stats for each column this format
            NB: String columns will NOT have num_zeros and descriptive_stats
                {'col1': {'dtype': 'string', 'unique': 4321, 'nulls': 12},
                 'col2': {'dtype': 'int', 'unique': 4321, 'nulls': 12, 'num_zeros': 100, 'descriptive_stats': {...}},
                 ...}
        """

        # Grab the column stats from our DataSource
        ds_column_stats = self.data_source.column_stats()

        # Map the types from our DataSource to the FeatureSet types
        fs_type_mapper = self.column_details()
        for col, details in ds_column_stats.items():
            details["fs_dtype"] = fs_type_mapper.get(col, "unknown")

        return ds_column_stats

    def ready(self) -> bool:
        """Is the FeatureSet ready? Is initial setup complete and expected metadata populated?
        Note: Since FeatureSet is a composite of DataSource and FeatureGroup, we need to
           check both to see if the FeatureSet is ready."""

        # Check if our parent class (Artifact) is ready
        if not super().ready():
            return False

        # Okay now call/return the DataSource ready() method
        return self.data_source.ready()

    def onboard(self) -> bool:
        """This is a BLOCKING method that will onboard the FeatureSet (make it ready)"""

        # Set our status to onboarding
        self.log.important(f"Onboarding {self.name}...")
        self.set_status("onboarding")
        self.remove_health_tag("needs_onboard")

        # Call our underlying DataSource onboard method
        self.data_source.refresh_meta()
        if not self.data_source.exists():
            self.log.critical(f"Data Source check failed for {self.name}")
            self.log.critical("Delete this Feature Set and recreate it to fix this issue")
            return False
        if not self.data_source.ready():
            self.data_source.onboard()

        # Run a health check and refresh the meta
        time.sleep(2)  # Give the AWS Metadata a chance to update
        self.health_check()
        self.refresh_meta()
        self.details()
        self.set_status("ready")
        return True

    def recompute_stats(self) -> bool:
        """This is a BLOCKING method that will recompute the stats for the FeatureSet"""

        # Call our underlying DataSource recompute stats method
        self.log.important(f"Recomputing Stats {self.name}...")
        self.data_source.recompute_stats()
        self.details()
        return True


if __name__ == "__main__":
    """Exercise for FeatureSet Class"""
    from workbench.core.artifacts.feature_set_core import FeatureSetCore as LocalFeatureSetCore  # noqa: F811
    from pprint import pprint

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Grab a FeatureSet object and pull some information from it
    my_features = LocalFeatureSetCore("abalone_features")
    if not my_features.exists():
        print("FeatureSet not found!")
        sys.exit(1)

    # Call the various methods
    # What's my AWS ARN and URL
    print(f"AWS ARN: {my_features.arn()}")
    print(f"AWS URL: {my_features.aws_url()}")

    # Let's do a check/validation of the feature set
    print(f"Feature Set Check: {my_features.exists()}")

    # How many rows and columns?
    num_rows = my_features.num_rows()
    num_columns = my_features.num_columns()
    print(f"Rows: {num_rows} Columns: {num_columns}")

    # What are the column names?
    print(my_features.columns)

    # Get the metadata and tags associated with this feature set
    print(f"Workbench Meta: {my_features.workbench_meta()}")
    print(f"Workbench Tags: {my_features.get_tags()}")

    # Get a summary for this Feature Set
    print("\nSummary:")
    pprint(my_features.summary())

    # Get the details for this Feature Set
    print("\nDetails:")
    pprint(my_features.details())

    # Now do deep dive on storage
    storage = my_features.get_data_source()
    print("\nStorage Details:")
    pprint(storage.details())

    # Test getting the holdout ids
    print("Getting the hold out ids...")
    holdout_ids = my_features.get_training_holdouts()
    print(f"Holdout IDs: {holdout_ids}")

    # Get a sample of the data
    df = my_features.sample()
    print(f"Sample Data: {df.shape}")
    print(df)

    # Get descriptive stats for all the columns
    stat_info = my_features.descriptive_stats()
    print("Descriptive Stats")
    pprint(stat_info)

    # Get outliers for all the columns
    outlier_df = my_features.outliers()
    print(outlier_df)

    # Set the holdout ids for the training view
    print("Setting hold out ids...")
    table = my_features.view("training").table
    df = my_features.query(f'SELECT auto_id, length FROM "{table}"')
    my_holdout_ids = [id for id in df["auto_id"] if id < 20]
    my_features.set_training_holdouts(my_holdout_ids)

    # Get the training data
    print("Getting the training data...")
    training_data = my_features.get_training_data()
    print(f"Training Data: {training_data.shape}")

    # Test the filter expression functionality
    print("Setting a filter expression...")
    my_features.set_training_filter("auto_id < 50 AND length > 65.0")
    training_data = my_features.get_training_data()
    print(f"Training Data: {training_data.shape}")
    print(training_data)

    # Remove training filter
    print("Removing the filter expression...")
    my_features.set_training_filter(None)
    training_data = my_features.get_training_data()
    print(f"Training Data: {training_data.shape}")
    print(training_data)

    # Test excluding ids from training
    print("Excluding ids from training...")
    my_features.exclude_ids_from_training([1, 2, 3, 4, 5])
    training_data = my_features.get_training_data()
    print(f"Training Data: {training_data.shape}")
    print(training_data)

    # Now delete the AWS artifacts associated with this Feature Set
    # print("Deleting Workbench Feature Set...")
    # my_features.delete()
    # print("Done")

    # Test set_training_sampling with exclusions and replications
    print("\n--- Testing set_training_sampling ---")
    my_features.set_training_filter(None)  # Reset any existing filters
    original_count = num_rows

    # Get valid IDs from the table
    all_data = my_features.query(f'SELECT auto_id, length FROM "{table}"')
    valid_ids = sorted(all_data["auto_id"].tolist())
    print(f"Valid IDs range from {valid_ids[0]} to {valid_ids[-1]}")

    exclude_list = valid_ids[0:3]  # First 3 IDs
    replicate_list = valid_ids[10:13]  # IDs at positions 10, 11, 12

    print(f"Original row count: {original_count}")
    print(f"Excluding IDs: {exclude_list}")
    print(f"Replicating IDs: {replicate_list}")

    # Test with default replication factor (2x)
    print("\n--- Testing with replication_factor=2 (default) ---")
    my_features.set_training_sampling(exclude_ids=exclude_list, replicate_ids=replicate_list)
    training_data = my_features.get_training_data()
    print(f"Training Data after sampling: {training_data.shape}")

    # Verify exclusions
    for exc_id in exclude_list:
        count = len(training_data[training_data["auto_id"] == exc_id])
        print(f"Excluded ID {exc_id} appears {count} times (should be 0)")

    # Verify replications
    for rep_id in replicate_list:
        count = len(training_data[training_data["auto_id"] == rep_id])
        print(f"Replicated ID {rep_id} appears {count} times (should be 2)")

    # Test with replication factor of 5
    print("\n--- Testing with replication_factor=5 ---")
    replicate_list_5x = [20, 21]
    my_features.set_training_sampling(exclude_ids=exclude_list, replicate_ids=replicate_list_5x, replication_factor=5)
    training_data = my_features.get_training_data()
    print(f"Training Data after sampling: {training_data.shape}")

    # Verify 5x replication
    for rep_id in replicate_list_5x:
        count = len(training_data[training_data["auto_id"] == rep_id])
        print(f"Replicated ID {rep_id} appears {count} times (should be 5)")

    # Test with large replication list (simulate 100 IDs)
    print("\n--- Testing with large ID list (100 IDs) ---")
    large_replicate_list = list(range(30, 130))  # 100 IDs
    my_features.set_training_sampling(replicate_ids=large_replicate_list, replication_factor=3)
    training_data = my_features.get_training_data()
    print(f"Training Data after sampling: {training_data.shape}")
    print(f"Expected extra rows: {len(large_replicate_list) * 3}")
