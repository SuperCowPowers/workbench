"""LocalFeatureSet: Engineered features on local disk, queryable with DuckDB."""

import os
from typing import Union

import duckdb
import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local.local_artifact import LocalArtifact
from workbench.local import storage
from workbench.utils import feature_prep


class LocalFeatureSet(LocalArtifact):
    """LocalFeatureSet: Workbench Local FeatureSet Class

    Common Usage:
        ```python
        my_features = LocalFeatureSet("my_features")
        my_features.query("select * from my_features where solubility < -5")
        my_model = my_features.to_model(...)
        ```
    """

    artifact_type = "feature_set"

    def __init__(self, name: str, **kwargs):
        """Initialize a LocalFeatureSet

        Args:
            name (str): The name of an existing local feature set
        """
        Artifact.is_name_valid(name)
        super().__init__(name, **kwargs)
        self.data_path = os.path.join(self.path, "data.parquet")

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        id_column: str,
        event_time_column: str = None,
        one_hot_columns: list = None,
        input_name: str = "dataframe",
    ) -> "LocalFeatureSet":
        """Create a LocalFeatureSet from a DataFrame, running the shared column prep.

        Args:
            df (pd.DataFrame): The DataFrame of features
            name (str): The name for the feature set (must be lowercase)
            id_column (str): The ID column (use "auto" for auto-generated IDs)
            event_time_column (str, optional): Event time column (default: None)
            one_hot_columns (list, optional): Columns to one-hot encode (default: None)
            input_name (str): Name of this feature set's input (default: "dataframe")

        Returns:
            LocalFeatureSet: The created feature set
        """
        # Same prep the AWS ingest path runs, so columns/names/dtypes match after publish
        df, id_column = feature_prep.prep_dataframe(
            df.copy(),
            id_column=id_column,
            event_time_column=event_time_column,
            one_hot_columns=one_hot_columns,
        )

        fs = cls(name)
        fs.log.important(f"Storing local feature set {name} ({len(df)} rows, {len(df.columns)} columns)...")
        storage.local_root(create=True)
        os.makedirs(fs.path, exist_ok=True)
        df.to_parquet(fs.data_path, index=False)
        fs._init_storage(input_name=input_name)
        fs.upsert_workbench_meta(
            {
                "id_column": id_column,
                "num_rows": len(df),
                "columns": list(df.columns),
                "column_types": [str(dtype) for dtype in df.dtypes],
            }
        )
        return fs

    @property
    def id_column(self) -> str:
        """The ID column for this FeatureSet"""
        return self.workbench_meta().get("id_column")

    @property
    def columns(self) -> list[str]:
        """Return the column names for this FeatureSet"""
        return list(self.workbench_meta().get("columns", []))

    @property
    def column_types(self) -> list[str]:
        """Return the column types for this FeatureSet"""
        return list(self.workbench_meta().get("column_types", []))

    def num_rows(self) -> int:
        """Return the number of rows for this FeatureSet"""
        return self.workbench_meta().get("num_rows", 0)

    def num_columns(self) -> int:
        """Return the number of columns for this FeatureSet"""
        return len(self.columns)

    def query(self, query: str) -> pd.DataFrame:
        """Query this FeatureSet with DuckDB

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
        """Return a DataFrame of ALL the data from this FeatureSet

        Args:
            limit (int): Limit the number of rows returned (default: None = all rows)

        Returns:
            pd.DataFrame: A DataFrame of the data from this FeatureSet
        """
        if not self.exists():
            self.log.error(f"Local artifact {self.name} does not exist...")
            return pd.DataFrame()
        df = pd.read_parquet(self.data_path)
        return df.head(limit) if limit else df

    def details(self, **kwargs) -> dict:
        """LocalFeatureSet Details

        Returns:
            dict: A dictionary of details about the LocalFeatureSet
        """
        return {
            **super().details(),
            "id_column": self.id_column,
            "num_rows": self.num_rows(),
            "num_columns": self.num_columns(),
        }

    def training_view(
        self,
        sample_weights: Union[dict, pd.DataFrame] = None,
        validation_ids: list = None,
        exclude_ids: list = None,
    ) -> pd.DataFrame:
        """Build the training frame: features plus the three role columns.

        Mirrors the AWS model training view: `sample_weight` (default 1.0),
        `validation` (default False), and `exclude` (default False). Excluded rows
        are dropped entirely, and exclude wins over validation.

        Args:
            sample_weights (Union[dict, pd.DataFrame], optional): id -> weight, forwarded as-is
            validation_ids (list, optional): ids held out of training and scored as a holdout
            exclude_ids (list, optional): ids dropped from the training frame entirely

        Returns:
            pd.DataFrame: The feature columns plus sample_weight/validation/exclude
        """
        df = self.pull_dataframe()
        ids = df[self.id_column]

        if isinstance(sample_weights, pd.DataFrame):
            sample_weights = dict(zip(sample_weights[self.id_column], sample_weights["sample_weight"]))
        df["sample_weight"] = ids.map(sample_weights).fillna(1.0) if sample_weights else 1.0
        df["validation"] = ids.isin(validation_ids) if validation_ids else False
        df["exclude"] = ids.isin(exclude_ids) if exclude_ids else False

        # Excluded rows never reach a model (exclude wins over validation)
        return df[~df["exclude"]].reset_index(drop=True)


if __name__ == "__main__":
    """Exercise the LocalFeatureSet Class"""
    from pprint import pprint

    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Height": [0.1, 0.4, 0.8, 0.6],
            "Food": ["steak", "tofu", "steak", "fish"],
            "Solubility": [-1.0, -3.0, -5.0, -2.0],
        }
    )
    fs = LocalFeatureSet.from_dataframe(df, name="local_test_features", id_column="ID", one_hot_columns=["Food"])
    pprint(fs.details())

    print(fs.query("select * from local_test_features where solubility < -2"))
    print(fs.training_view(validation_ids=[3], exclude_ids=[4]))
