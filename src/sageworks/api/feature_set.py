"""FeatureSet: Manages AWS Feature Store/Group creation and management.
FeatureSets are set up so they can easily be queried with AWS Athena.
All FeatureSets are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) FeatureSets
can be viewed and explored within the SageWorks Dashboard UI."""

from typing import Union
import pandas as pd

# SageWorks Imports
from sageworks.core.artifacts.artifact import Artifact
from sageworks.core.artifacts.feature_set_core import FeatureSetCore
from sageworks.core.transforms.features_to_model.features_to_model import FeaturesToModel
from sageworks.api.model import Model, ModelType


class FeatureSet(FeatureSetCore):
    """FeatureSet: SageWorks FeatureSet API Class

    Common Usage:
        ```python
        my_features = FeatureSet(name)
        my_features.details()
        my_features.to_model(
            ModelType.REGRESSOR,
            name="abalone-regression",
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

    def pull_dataframe(self, include_aws_columns=False) -> pd.DataFrame:
        """Return a DataFrame of ALL the data from this FeatureSet

        Args:
            include_aws_columns (bool): Include the AWS columns in the DataFrame (default: False)

        Returns:
            pd.DataFrame: A DataFrame of ALL the data from this FeatureSet

        Note:
            Obviously this is not recommended for large datasets :)
        """

        # Get the table associated with the data
        self.log.info(f"Pulling all data from {self.uuid}...")
        query = f"SELECT * FROM {self.athena_table}"
        df = self.query(query)

        # Drop any columns generated from AWS
        if not include_aws_columns:
            aws_cols = ["write_time", "api_invocation_time", "is_deleted", "event_time"]
            df = df.drop(columns=aws_cols, errors="ignore")
        return df

    def to_model(
        self,
        model_type: ModelType = ModelType.UNKNOWN,
        model_class: str = None,
        name: str = None,
        tags: list = None,
        description: str = None,
        feature_list: list = None,
        target_column: str = None,
        **kwargs,
    ) -> Union[Model, None]:
        """Create a Model from the FeatureSet

        Args:

            model_type (ModelType): The type of model to create (See sageworks.model.ModelType)
            model_class (str): The model class to use for the model (e.g. "KNeighborsRegressor", default: None)
            name (str): Set the name for the model. If not specified, a name will be generated
            tags (list): Set the tags for the model.  If not specified tags will be generated.
            description (str): Set the description for the model. If not specified a description is generated.
            feature_list (list): Set the feature list for the model. If not specified a feature list is generated.
            target_column (str): The target column for the model (use None for unsupervised model)

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
            name = self.uuid.replace("_features", "") + "-model"
            name = Artifact.generate_valid_name(name, delimiter="-")

        # Create the Model Tags
        tags = [name] if tags is None else tags

        # Transform the FeatureSet into a Model
        features_to_model = FeaturesToModel(self.uuid, name, model_type=model_type, model_class=model_class)
        features_to_model.set_output_tags(tags)
        features_to_model.transform(
            target_column=target_column, description=description, feature_list=feature_list, **kwargs
        )

        # Return the Model
        return Model(name)


if __name__ == "__main__":
    """Exercise the FeatureSet Class"""
    from pprint import pprint

    # Retrieve an existing FeatureSet
    my_features = FeatureSet("test_features")
    pprint(my_features.summary())
    pprint(my_features.details())

    # Create a Model from the FeatureSet
    my_model = my_features.to_model(model_type=ModelType.REGRESSOR, target_column="iq_score")
