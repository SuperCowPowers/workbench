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

    def pull_dataframe(self, limit: int = 50000, include_aws_columns=False) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this FeatureSet

        Args:
            limit (int): Limit the number of rows returned (default: 50000)
            include_aws_columns (bool): Include the AWS columns in the DataFrame (default: False)

        Returns:
            pd.DataFrame: A DataFrame of all the data from this FeatureSet up to the limit
        """

        # Get the table associated with the data
        self.log.info(f"Pulling data from {self.name}...")
        pull_query = f'SELECT * FROM "{self.athena_table}" LIMIT {limit}'
        df = self.query(pull_query)

        # Drop any columns generated from AWS
        if not include_aws_columns:
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
            df = df.drop(columns=aws_cols, errors="ignore")
        return df

    def to_model(
        self,
        name: str,
        model_type: ModelType,
        model_framework: ModelFramework = ModelFramework.XGBOOST,
        tags: list = None,
        description: str = None,
        feature_list: list = None,
        target_column: Union[str, list[str]] = None,
        model_class: str = None,
        model_import_str: str = None,
        custom_script: Union[str, Path] = None,
        custom_args: dict = None,
        training_image: str = "training",
        inference_image: str = "inference",
        inference_arch: str = "x86_64",
        **kwargs,
    ) -> Union[Model, None]:
        """Create a Model from the FeatureSet

        Args:

            name (str): The name of the Model to create
            model_type (ModelType): The type of model to create (See workbench.model.ModelType)
            model_framework (ModelFramework, optional): The framework to use for the model (default: XGBOOST)
            tags (list, optional): Set the tags for the model.  If not given tags will be generated.
            description (str, optional): Set the description for the model. If not give a description is generated.
            feature_list (list, optional): Set the feature list for the model. If not given a feature list is generated.
            target_column (str or list[str], optional): Target column(s) for the model (use None for unsupervised model)
            model_class (str, optional): Model class to use (e.g. "KMeans", default: None)
            model_import_str (str, optional): The import for the model (e.g. "from sklearn.cluster import KMeans")
            custom_script (str, optional): The custom script to use for the model (default: None)
            training_image (str, optional): The training image to use (default: "training")
            inference_image (str, optional): The inference image to use (default: "inference")
            inference_arch (str, optional): The architecture to use for inference (default: "x86_64")
            kwargs (dict, optional): Additional keyword arguments to pass to the model

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

        # If the model framework is PyTorch or ChemProp, ensure we set the training and inference images
        if model_framework in (ModelFramework.PYTORCH, ModelFramework.CHEMPROP):
            training_image = "pytorch_training"
            inference_image = "pytorch_inference"

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

    def prox_model(
        self, target: str, features: list, include_all_columns: bool = False
    ) -> "FeatureSpaceProximity":  # noqa: F821
        """Create a local FeatureSpaceProximity Model for this FeatureSet

        Args:
           target (str): The target column name
           features (list): The list of feature column names
           include_all_columns (bool): Include all DataFrame columns in results (default: False)

        Returns:
           FeatureSpaceProximity: A local FeatureSpaceProximity Model
        """
        from workbench.algorithms.dataframe.feature_space_proximity import FeatureSpaceProximity  # noqa: F401

        # Create the Proximity Model from the full FeatureSet dataframe
        full_df = self.pull_dataframe()

        # Create and return the FeatureSpaceProximity Model
        return FeatureSpaceProximity(
            full_df, id_column=self.id_column, features=features, target=target, include_all_columns=include_all_columns
        )

    def fp_prox_model(
        self,
        target: str,
        fingerprint_column: str = None,
        include_all_columns: bool = False,
        radius: int = 2,
        n_bits: int = 1024,
        counts: bool = False,
    ) -> "FingerprintProximity":  # noqa: F821
        """Create a local FingerprintProximity Model for this FeatureSet

        Args:
           target (str): The target column name
           fingerprint_column (str): Column containing fingerprints. If None, uses existing 'fingerprint'
                                     column or computes from SMILES column.
           include_all_columns (bool): Include all DataFrame columns in results (default: False)
           radius (int): Radius for Morgan fingerprint computation (default: 2)
           n_bits (int): Number of bits for fingerprint (default: 1024)
           counts (bool): Whether to use count simulation (default: False)

        Returns:
           FingerprintProximity: A local FingerprintProximity Model
        """
        from workbench.algorithms.dataframe.fingerprint_proximity import FingerprintProximity  # noqa: F401

        # Create the Proximity Model from the full FeatureSet dataframe
        full_df = self.pull_dataframe()

        # Create and return the FingerprintProximity Model
        return FingerprintProximity(
            full_df,
            id_column=self.id_column,
            fingerprint_column=fingerprint_column,
            target=target,
            include_all_columns=include_all_columns,
            radius=radius,
            n_bits=n_bits,
        )

    def cleanlab_model(
        self,
        target: str,
        features: list,
        model_type: ModelType = ModelType.REGRESSOR,
    ) -> "CleanLearning":  # noqa: F821
        """Create a CleanLearning model for detecting label issues in this FeatureSet

        Args:
           target (str): The target column name
           features (list): The list of feature column names
           model_type (ModelType): The model type (REGRESSOR or CLASSIFIER). Defaults to REGRESSOR.

        Returns:
           CleanLearning: A fitted cleanlab model. Use get_label_issues() to get
           a DataFrame with id_column, label_quality, predicted_label, given_label, is_label_issue.
        """
        from workbench.algorithms.models.cleanlab_model import create_cleanlab_model  # noqa: F401

        # Get the full FeatureSet dataframe
        full_df = self.pull_dataframe()

        # Create and return the CleanLearning model
        return create_cleanlab_model(full_df, self.id_column, features, target, model_type=model_type)


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
    my_prox = my_features.prox_model(target="salary", features=features)
    neighbors = my_prox.neighbors(42)
    print("Neighbors for ID 42:")
    print(neighbors)

    # Create a Model from the FeatureSet
    """
    my_model = my_features.to_model(
        name="test-model",
        model_type=ModelType.REGRESSOR,
        target_column="salary",
        feature_list=features
    )
    pprint(my_model.summary())
    """
