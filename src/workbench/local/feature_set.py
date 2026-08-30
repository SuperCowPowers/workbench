"""FeatureSet: Engineered features on local disk, queryable with DuckDB."""

import os
from typing import Any, Union

import duckdb
import pandas as pd

# Workbench Imports
from workbench.core.artifact import Artifact
from workbench.local.local_artifact import LocalArtifact
from workbench.local import storage
from workbench.utils import feature_prep


class FeatureSet(LocalArtifact):
    """FeatureSet: Workbench Local FeatureSet Class

    Common Usage:
        ```python
        my_features = FeatureSet("my_features")
        my_features.query("select * from my_features where solubility < -5")
        my_model = my_features.to_model(...)
        ```
    """

    artifact_type = "feature_set"

    def __init__(self, name: str, **kwargs):
        """Initialize a FeatureSet

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
    ) -> "FeatureSet":
        """Create a FeatureSet from a DataFrame, running the shared column prep.

        Args:
            df (pd.DataFrame): The DataFrame of features
            name (str): The name for the feature set (must be lowercase)
            id_column (str): The ID column (use "auto" for auto-generated IDs)
            event_time_column (str, optional): Event time column (default: None)
            one_hot_columns (list, optional): Columns to one-hot encode (default: None)
            input_name (str): Name of this feature set's input (default: "dataframe")

        Returns:
            FeatureSet: The created feature set
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

    def prox(
        self,
        space: str,
        feature_list: list = None,
        target: str = None,
        include_all_columns: bool = False,
    ) -> "Union[FingerprintProximity, FeatureSpaceProximity]":  # noqa: F821
        """Create (or reuse) a proximity model over this FeatureSet.

        For finding issues/anomalies or nearest neighbors before building a model.
        Cached per ``(space, feature_list, target)`` on this instance, so repeated
        calls return the same model.

        Args:
            space: ``"fingerprint"`` (Tanimoto over SMILES/fingerprints) or
                ``"features"`` (standardized Euclidean over numeric features).
            feature_list: Numeric columns for neighbor computation. Required for
                ``space="features"``; ignored for ``"fingerprint"``.
            target: Target column surfaced in neighbor results (optional).
            include_all_columns: Include all DataFrame columns in neighbor results.

        Returns:
            FingerprintProximity or FeatureSpaceProximity.
        """
        from workbench.utils.prox_utils import build_proximity

        key = (space, tuple(feature_list) if feature_list else None, target)
        if not hasattr(self, "_prox_cache"):
            self._prox_cache = {}
        if key not in self._prox_cache:
            self._prox_cache[key] = build_proximity(
                self.pull_dataframe(),
                space,
                self.id_column,
                feature_list=feature_list,
                target=target,
                include_all_columns=include_all_columns,
            )
        return self._prox_cache[key]

    def details(self, **kwargs) -> dict:
        """FeatureSet Details

        Returns:
            dict: A dictionary of details about the FeatureSet
        """
        return {
            **super().details(),
            "id_column": self.id_column,
            "num_rows": self.num_rows(),
            "num_columns": self.num_columns(),
        }

    def to_model(self, name: str, model_type, model_framework, **kwargs: Any) -> "Model":  # noqa: F821
        """Train a Model from this FeatureSet.

        Args:
            name (str): The name of the Model to create
            model_type (ModelType): The type of model to create
            model_framework (ModelFramework): The framework to use
            **kwargs: Passed to Model.from_feature_set (target_column, feature_list,
                hyperparameters, sample_weights, validation_ids, exclude_ids, wait)

        Returns:
            Model: The Model created from this FeatureSet
        """
        from workbench.local.model import Model

        return Model.from_feature_set(self, name=name, model_type=model_type, model_framework=model_framework, **kwargs)

    def parent(self):
        """The DataSource this FeatureSet came from, if it still exists locally"""
        from workbench.local.data_source import DataSource

        source = DataSource(self.get_input())
        return source if source.exists() else None

    def aws_exists(self) -> bool:
        """Does an AWS FeatureSet by this name already exist?

        Returns:
            bool: True if AWS already has this FeatureSet
        """
        from workbench.api import FeatureSet as AWSFeatureSet

        return AWSFeatureSet(self.name).exists()

    def _aws_artifact(self):
        """Internal: The AWS FeatureSet for this local one"""
        from workbench.api import FeatureSet as AWSFeatureSet

        return AWSFeatureSet(self.name)

    def _publish_self(self, **kwargs):
        """Internal: Push this FeatureSet's engineered features to AWS.

        The local frame is published as-is rather than recomputing features from the
        AWS DataSource, so the published feature values are exactly the ones trained on.

        Returns:
            FeatureSet: The created AWS FeatureSet
        """
        from workbench.api import FeatureSet as AWSFeatureSet
        from workbench.core.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

        to_features = PandasToFeatures(self.name)
        to_features.set_input(self.pull_dataframe(), id_column=self.id_column)
        to_features.set_output_tags([self.name])
        to_features.transform()
        return AWSFeatureSet(self.name)

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
    """Exercise the FeatureSet Class"""
    from pprint import pprint

    df = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Height": [0.1, 0.4, 0.8, 0.6],
            "Food": ["steak", "tofu", "steak", "fish"],
            "Solubility": [-1.0, -3.0, -5.0, -2.0],
        }
    )
    fs = FeatureSet.from_dataframe(df, name="local_test_features", id_column="ID", one_hot_columns=["Food"])
    pprint(fs.details())

    print(fs.query("select * from local_test_features where solubility < -2"))
    print(fs.training_view(validation_ids=[3], exclude_ids=[4]))
