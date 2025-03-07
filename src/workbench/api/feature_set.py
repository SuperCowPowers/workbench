"""FeatureSet: Manages AWS Feature Store/Group creation and management.
FeatureSets are set up so they can easily be queried with AWS Athena.
All FeatureSets are run through a full set of Exploratory Data Analysis (EDA)
techniques (data quality, distributions, stats, outliers, etc.) FeatureSets
can be viewed and explored within the Workbench Dashboard UI."""

from typing import Union
import pandas as pd

# Workbench Imports
from workbench.core.artifacts.artifact import Artifact
from workbench.core.artifacts.feature_set_core import FeatureSetCore
from workbench.core.transforms.features_to_model.features_to_model import FeaturesToModel
from workbench.api.model import Model, ModelType


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
            pd.DataFrame: A DataFrame of ALL the data from this FeatureSet

        Note:
            Obviously this is not recommended for large datasets :)
        """

        # Get the table associated with the data
        self.log.info(f"Pulling data from {self.uuid}...")
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
        tags: list = None,
        description: str = None,
        feature_list: list = None,
        target_column: str = None,
        scikit_model_class: str = None,
        model_import_str: str = None,
        custom_script: str = None,
        inference_arch: str = "x86_64",
        **kwargs,
    ) -> Union[Model, None]:
        """Create a Model from the FeatureSet

        Args:

            name (str): The name of the Model to create
            model_type (ModelType): The type of model to create (See workbench.model.ModelType)
            tags (list, optional): Set the tags for the model.  If not given tags will be generated.
            description (str, optional): Set the description for the model. If not give a description is generated.
            feature_list (list, optional): Set the feature list for the model. If not given a feature list is generated.
            target_column (str, optional): The target column for the model (use None for unsupervised model)
            scikit_model_class (str, optional): Scikit model class to use (e.g. "KMeans", default: None)
            model_import_str (str, optional): The import for the model (e.g. "from sklearn.cluster import KMeans")
            custom_script (str, optional): The custom script to use for the model (default: None)
            inference_arch (str, optional): The architecture to use for inference (default: "x86_64")

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
        features_to_model = FeaturesToModel(
            feature_uuid=self.uuid,
            model_uuid=name,
            model_type=model_type,
            scikit_model_class=scikit_model_class,
            model_import_str=model_import_str,
            custom_script=custom_script,
            inference_arch=inference_arch,
        )
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
    my_model = my_features.to_model(name="test-model", model_type=ModelType.REGRESSOR, target_column="iq_score")
