"""FeatureSet: Manages AWS Feature Store/Group creation and management.
FeatureSets are set up so they can easily be queried with AWS Athena.
All FeatureSets are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) FeatureSets
can be viewed and explored within the SageWorks Dashboard UI."""

import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.api.model import Model, ModelType


class FeatureSet(FeatureSetCore):
    """FeatureSet: SageWorks FeatureSet API Class

    Common Usage:
        ```
        my_features = FeatureSet(name)
        my_features.details()
        my_features.to_model(
            ModelType.REGRESSOR,
            name="abalone-regression",
            target_column="class_number_of_rings"
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

    def pull_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this FeatureSet

        Returns:
            pd.DataFrame: A DataFrame of ALL the data from this FeatureSet

        Note:
            Obviously, this is not recommended for large datasets :)
        """

        # Get the table associated with the data
        self.log.info(f"Pulling all data from {self.uuid}...")
        query = f"SELECT * FROM {self.athena_table}"
        return self.query(query)

    def to_model(
        self,
        model_type: ModelType,
        target_column: str,
        name: str = None,
        tags: list = None,
        description: str = None,
        feature_list: list = None,
        **kwargs,
    ) -> Model:
        """Create a Model from the FeatureSet

        Args:
            model_type (ModelType): The type of model to create (See sageworks.model.ModelType)
            target_column (str): The target column for the model (use None for unsupervised model)
            name (str): Set the name for the model. If not specified, a name will be generated
            tags (list): Set the tags for the model.  If not specified tags will be generated.
            description (str): Set the description for the model. If not specified a description is generated.
            feature_list (list): Set the feature list for the model. If not specified a feature list is generated.

        Returns:
            Model: The Model created from the FeatureSet
        """

        # Create the Model Name and Tags
        model_name = self.uuid.replace("_features", "").replace("_", "-") + "-model" if name is None else name
        model_name = Artifact.base_compliant_uuid(model_name, delimiter="-")

        # Create the Model Tags
        tags = [model_name] if tags is None else tags

        # Transform the FeatureSet into a Model
        features_to_model = FeaturesToModel(self.uuid, model_name, model_type=model_type)
        features_to_model.set_output_tags(tags)
        features_to_model.transform(
            target_column=target_column, description=description, feature_list=feature_list, **kwargs
        )

        # Return the Model
        return Model(model_name)


if __name__ == "__main__":
    """Exercise the FeatureSet Class"""
    from pprint import pprint

    # Retrieve an existing FeatureSet
    my_features = FeatureSet("test_features")
    pprint(my_features.summary())
    pprint(my_features.details())

    # Create a Model from the FeatureSet
    my_model = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="iq_score")
