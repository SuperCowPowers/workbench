"""FeatureSet: Manages AWS Feature Store/Group creation and management.
FeatureSets are set up so they can easily be queried with AWS Athena.
All FeatureSets are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) FeatureSets
can be viewed and explored within the Workbench Dashboard UI."""

from typing import Union
from pathlib import Path
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
from workbench.api.model import Model, ModelType, ModelFramework


class FeatureSet(FeatureSetCore):
    """FeatureSet: Workbench FeatureSet API Class

    Common Usage:
        ```python
        my_features = FeatureSet(name)
        my_features.details()
        my_features.to_model(
            name="abalone-regression",
            model_type=ModelType.REGRESSOR,
            model_framework=ModelFramework.XGBOOST,
            target_column="class_number_of_rings"
            feature_list=["my", "best", "features"])
        )
        ```
    """

    def details(self, **kwargs) -> dict:
        """FeatureSet Details

        Returns:
            dict: A dictionary of details about the FeatureSet
        """
        return super().details(**kwargs)

    def query(self, query: str, **kwargs) -> pd.DataFrame:
        """Query the AthenaSource

        Args:
            query (str): The query to run against the FeatureSet

        Returns:
            pd.DataFrame: The results of the query
        """
        return super().query(query, **kwargs)

    def pull_dataframe(self, limit: int = 100000, include_aws_columns=False) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this FeatureSet

        Args:
            limit (int): Limit the number of rows returned (default: 100000)
            include_aws_columns (bool): Include the AWS columns in the DataFrame (default: False)

        Returns:
            pd.DataFrame: A DataFrame of all the data from this FeatureSet up to the limit
        """
        return super().pull_dataframe(limit, include_aws_columns)

    def to_model(
        self,
        name: str,
        model_type: ModelType,
        model_framework: ModelFramework,
        tags: list = None,
        description: str = None,
        feature_list: list = None,
        target_column: Union[str, list[str]] = None,
        model_class: str = None,
        model_import_str: str = None,
        custom_script: Union[str, Path] = None,
        custom_args: dict = None,
        **kwargs,
    ) -> Union[Model, None]:
        """Create a Model from the FeatureSet

        Args:

            name (str): The name of the Model to create
            model_type (ModelType): The type of model to create (See workbench.model.ModelType)
            model_framework (ModelFramework): The framework (SKLEARN, XGBOOST, PYTORCH, CHEMPROP, TRANSFORMER, etc.)
            tags (list, optional): Set the tags for the model.  If not given tags will be generated.
            description (str, optional): Set the description for the model. If not give a description is generated.
            feature_list (list, optional): Set the feature list for the model. If not given a feature list is generated.
            target_column (str or list[str], optional): Target column(s) for the model (use None for unsupervised model)
            model_class (str, optional): Model class to use (e.g. "KMeans", default: None)
            model_import_str (str, optional): The import for the model (e.g. "from sklearn.cluster import KMeans")
            custom_script (str, optional): The custom script to use for the model (default: None)
            kwargs (dict, optional): Additional keyword arguments to pass to the model. Notably:
                ``sample_weights`` (dict|DataFrame): pure per-id framework weight, forwarded as-is;
                ``validation_ids`` (list): ids held out as a scored in-training validation set;
                ``exclude_ids`` (list): ids dropped from the training view entirely.

        Returns:
            Model: The Model created from the FeatureSet (or None if the Model could not be created)
        """

        # Ensure the model_name is valid
        if name:
            if not Artifact.is_name_valid(name, delimiter="-", lower_case=False):
                self.log.critical(f"Invalid Model name: {name}, not creating Model!")
                return None

        # If the model_name wasn't given generate it
        else:
            name = self.name.replace("_features", "") + "-model"
            name = Artifact.generate_valid_name(name, delimiter="-")

        # Create the Model Tags
        tags = [name] if tags is None else tags

        # Set training/inference images based on model framework
        if model_framework in (ModelFramework.PYTORCH, ModelFramework.CHEMPROP):
            training_image = "pytorch_chem_training"
            inference_image = "pytorch_chem_inference"
            inference_arch = "x86_64"
        else:
            training_image = "base_training"
            inference_image = "base_inference"
            inference_arch = "x86_64"

        # Transform the FeatureSet into a Model
        features_to_model = FeaturesToModel(
            feature_name=self.name,
            model_name=name,
            model_type=model_type,
            model_framework=model_framework,
            model_class=model_class,
            model_import_str=model_import_str,
            custom_script=custom_script,
            custom_args=custom_args,
            training_image=training_image,
            inference_image=inference_image,
            inference_arch=inference_arch,
        )
        features_to_model.set_output_tags(tags)
        features_to_model.transform(
            target_column=target_column, description=description, feature_list=feature_list, **kwargs
        )

        # Return the Model
        return Model(name)

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
        if space not in ("fingerprint", "features"):
            raise ValueError(f"space must be 'fingerprint' or 'feature', got {space!r}")
        if space == "features" and not feature_list:
            raise ValueError("space='feature' requires feature_list=[...]")

        key = (space, tuple(feature_list) if feature_list else None, target)
        if not hasattr(self, "_prox_cache"):
            self._prox_cache = {}
        if key in self._prox_cache:
            return self._prox_cache[key]

        full_df = self.pull_dataframe()
        if space == "fingerprint":
            from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity

            prox = FingerprintProximity(
                full_df, id_column=self.id_column, target=target, include_all_columns=include_all_columns
            )
        else:
            from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity

            prox = FeatureSpaceProximity(
                full_df,
                id_column=self.id_column,
                features=feature_list,
                target=target,
                include_all_columns=include_all_columns,
            )

        self._prox_cache[key] = prox
        return prox

    def cleanlab_model(
        self,
        target: str,
        features: list,
        model_type: ModelType = ModelType.REGRESSOR,
    ) -> "CleanlabModels":  # noqa: F821
        """Create a CleanlabModels instance for label-quality analysis of this FeatureSet

        Args:
           target (str): The target column name
           features (list): The list of feature column names
           model_type (ModelType): The model type (REGRESSOR or CLASSIFIER). Defaults to REGRESSOR.

        Returns:
           CleanlabModels: Label-quality analysis with helpers like label_issues()
           (a DataFrame keyed by id_column, sorted by label_quality) and the native
           clean_learning()/datalab() objects.
        """
        from workbench.algorithms.models.cleanlab_model import CleanlabModels  # noqa: F401

        # Get the full FeatureSet dataframe
        full_df = self.pull_dataframe()

        # Create and return the CleanlabModels instance
        return CleanlabModels(full_df, self.id_column, features, target, model_type=model_type)


if __name__ == "__main__":
    """Exercise the FeatureSet Class"""
    from pprint import pprint

    # Retrieve an existing FeatureSet
    my_features = FeatureSet("test_features")
    pprint(my_features.summary())
    pprint(my_features.details())

    # Pull the full DataFrame
    df = my_features.pull_dataframe()
    print(df.head())

    # Create a Proximity Model from the FeatureSet
    features = ["height", "weight", "age", "iq_score", "likes_dogs", "food"]
    my_prox = my_features.prox("features", feature_list=features, target="salary")
    neighbors = my_prox.neighbors(42)
    print("Neighbors for ID 42:")
    print(neighbors)

    # Create a Model from the FeatureSet
    """
    my_model = my_features.to_model(
        name="test-model",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.XGBOOST,
        target_column="salary",
        feature_list=features
    )
    pprint(my_model.summary())
    """
