"""LocalDataSource: A DataFrame on local disk, queryable with DuckDB."""

import os
from typing import Union

import duckdb
import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local.local_artifact import LocalArtifact
from workbench.local import storage


class LocalDataSource(LocalArtifact):
    """LocalDataSource: Workbench Local DataSource Class

    Common Usage:
        ```python
        my_data = LocalDataSource(df, name="my_data")
        my_data.query("select * from my_data where height > 0.3")
        my_features = my_data.to_features("my_features", id_column="id")
        ```
    """

    artifact_type = "data_source"

    def __init__(self, source: Union[str, pd.DataFrame] = None, name: str = None, **kwargs):
        """Initialize a LocalDataSource

        Args:
            source (Union[str, pd.DataFrame]): A DataFrame, a CSV/parquet file path, or an
                existing LocalDataSource name. If None, `name` must reference an existing source.
            name (str): The name of the data source (must be lowercase). Required for DataFrames.
        """
        # A bare name refers to an existing local data source
        if isinstance(source, str) and name is None and not os.path.isfile(source):
            name = source
            source = None

        # Derive a name from a file path when one wasn't given
        if name is None and isinstance(source, str):
            name = Artifact.generate_valid_name(os.path.splitext(os.path.basename(source))[0])

        if name is None:
            msg = "Set the 'name' argument: LocalDataSource(df, name='my_data')"
            self.log.critical(msg)
            raise ValueError(msg)
        Artifact.is_name_valid(name)

        # Call superclass init (sets up paths)
        super().__init__(name, **kwargs)
        self.data_path = os.path.join(self.path, "data.parquet")

        # Load the source (if given)
        if source is not None:
            self._load_source(source)

    def query(self, query: str) -> pd.DataFrame:
        """Query this DataSource with DuckDB

        Args:
            query (str): SQL to run; reference this artifact by its name

        Returns:
            pd.DataFrame: The results of the query
        """
        if not self.exists():
            self.log.error(f"Local artifact {self.name} does not exist...")
            return pd.DataFrame()

        with duckdb.connect() as con:
            con.execute(f"CREATE VIEW \"{self.name}\" AS SELECT * FROM read_parquet('{self.data_path}')")
            return con.execute(query).df()

    def pull_dataframe(self, limit: int = None) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this DataSource

        Args:
            limit (int): Limit the number of rows returned (default: None = all rows)

        Returns:
            pd.DataFrame: A DataFrame of the data from this DataSource
        """
        if not self.exists():
            self.log.error(f"Local artifact {self.name} does not exist...")
            return pd.DataFrame()
        df = pd.read_parquet(self.data_path)
        return df.head(limit) if limit else df

    @property
    def columns(self) -> list[str]:
        """Return the column names for this DataSource"""
        return list(self.workbench_meta().get("columns", []))

    @property
    def column_types(self) -> list[str]:
        """Return the column types for this DataSource"""
        return list(self.workbench_meta().get("column_types", []))

    def num_rows(self) -> int:
        """Return the number of rows for this DataSource"""
        return self.workbench_meta().get("num_rows", 0)

    def num_columns(self) -> int:
        """Return the number of columns for this DataSource"""
        return len(self.columns)

    def details(self, **kwargs) -> dict:
        """LocalDataSource Details

        Returns:
            dict: A dictionary of details about the LocalDataSource
        """
        return {**super().details(), "num_rows": self.num_rows(), "num_columns": self.num_columns()}

    def to_features(
        self,
        name: str,
        id_column: str,
        tags: list = None,
        event_time_column: str = None,
        one_hot_columns: list = None,
    ) -> Union["LocalFeatureSet", None]:  # noqa: F821
        """Convert this LocalDataSource to a LocalFeatureSet

        Args:
            name (str): Set the name for the feature set (must be lowercase).
            id_column (str): The ID column (must be specified, use "auto" for auto-generated IDs).
            tags (list, optional): Set the tags for the feature set (unused, kept for API parity).
            event_time_column (str, optional): The event time column (default: None).
            one_hot_columns (list, optional): Columns to one-hot encode (default: None).

        Returns:
            LocalFeatureSet: The FeatureSet created from this DataSource (or None on invalid name)
        """
        from workbench.local.local_feature_set import LocalFeatureSet

        if not Artifact.is_name_valid(name):
            self.log.critical(f"Invalid FeatureSet name: {name}, not creating FeatureSet!")
            return None

        return LocalFeatureSet.from_dataframe(
            self.pull_dataframe(),
            name=name,
            id_column=id_column,
            event_time_column=event_time_column,
            one_hot_columns=one_hot_columns,
            input_name=self.name,
        )

    def aws_exists(self) -> bool:
        """Does an AWS DataSource by this name already exist?

        Returns:
            bool: True if AWS already has this DataSource
        """
        from workbench.api import DataSource

        return DataSource(self.name).exists()

    def _aws_artifact(self):
        """Internal: The AWS DataSource for this local one"""
        from workbench.api import DataSource

        return DataSource(self.name)

    def _publish_self(self, **kwargs):
        """Internal: Create the AWS DataSource from this local one

        Returns:
            DataSource: The created AWS DataSource
        """
        from workbench.api import DataSource

        return DataSource(self.pull_dataframe(), name=self.name)

    def _load_source(self, source: Union[str, pd.DataFrame]):
        """Internal: Write the source data to local storage

        Args:
            source (Union[str, pd.DataFrame]): A DataFrame or a CSV/parquet file path
        """
        if isinstance(source, pd.DataFrame):
            df = source
        elif source.endswith(".parquet"):
            df = pd.read_parquet(source)
        else:
            df = pd.read_csv(source)

        self.log.important(f"Storing local data source {self.name} ({len(df)} rows)...")
        storage.local_root(create=True)
        os.makedirs(self.path, exist_ok=True)
        df.to_parquet(self.data_path, index=False)
        self._init_storage(input_name="dataframe" if isinstance(source, pd.DataFrame) else str(source))
        self.upsert_workbench_meta(
            {
                "num_rows": len(df),
                "columns": list(df.columns),
                "column_types": [str(dtype) for dtype in df.dtypes],
            }
        )


if __name__ == "__main__":
    """Exercise the LocalDataSource Class"""
    from pprint import pprint

    df = pd.DataFrame({"id": [1, 2, 3], "height": [0.1, 0.4, 0.8], "name": ["a", "b", "c"]})
    my_data = LocalDataSource(df, name="local_test_data")
    pprint(my_data.details())

    print(my_data.query("select * from local_test_data where height > 0.3"))
    print(my_data.pull_dataframe())

    my_features = my_data.to_features("local_test_features", id_column="id")
    pprint(my_features.details())
