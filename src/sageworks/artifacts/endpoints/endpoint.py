"""Endpoint: SageWorks Endpoint Class"""
import sys
from datetime import datetime
import botocore
import pandas as pd
import numpy as np
from io import StringIO

# Model Performance Scores
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, precision_recall_fscore_support
from math import sqrt

from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor

# SageWorks Imports
from sageworks.artifacts.artifact import Artifact
from sageworks.aws_service_broker.aws_service_broker import ServiceCategory


class Endpoint(Artifact):
    """Endpoint: SageWorks Endpoint Class

    Common Usage:
        my_endpoint = Endpoint(endpoint_uuid)
        prediction_df = my_endpoint.predict(test_df)
        metrics = my_endpoint.regression_metrics(target_column, prediction_df)
        for metric, value in metrics.items():
            print(f"{metric}: {value:0.3f}")
    """

    def __init__(self, endpoint_uuid, force_refresh: bool = False, exit_on_error=True):
        """Endpoint Initialization

        Args:
            endpoint_uuid (str): Name of Endpoint in SageWorks
            force_refresh (bool, optional): Force a refresh of the AWS Broker. Defaults to False.
        """
        # Call SuperClass Initialization
        super().__init__(endpoint_uuid)

        # Grab an AWS Metadata Broker object and pull information for Endpoints
        self.endpoint_name = endpoint_uuid
        self.endpoint_meta = self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS, force_refresh=force_refresh).get(
            self.endpoint_name
        )
        self.endpoint_return_columns = None
        self.exit_on_error = exit_on_error

        # All done
        self.log.info(f"Endpoint Initialized: {self.endpoint_name}")

    def refresh_meta(self):
        """Refresh the Artifact's metadata"""
        self.endpoint_meta = self.aws_broker.get_metadata(ServiceCategory.ENDPOINTS, force_refresh=True).get(
            self.endpoint_name
        )

    def exists(self) -> bool:
        """Does the feature_set_name exist in the AWS Metadata?"""
        if self.endpoint_meta is None:
            self.log.info(f"Endpoint {self.endpoint_name} not found in AWS Metadata!")
            return False
        return True

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference/prediction on the given Feature DataFrame
        Args:
            feature_df (pd.DataFrame): DataFrame to run predictions on (must have superset of features)
        Returns:
            pd.DataFrame: Return the feature DataFrame with two additional columns (prediction, pred_proba)
        """

        # Create our Endpoint Predictor Class
        predictor = Predictor(
            self.endpoint_name,
            sagemaker_session=self.sm_session,
            serializer=CSVSerializer(),
            deserializer=CSVDeserializer(),
        )

        # Now split up the dataframe into 500 row chunks, send those chunks to our
        # endpoint (with error handling) and stitch all the chunks back together
        df_list = []
        for index in range(0, len(feature_df), 500):
            print("Processing...")

            # Compute partial DataFrames, add them to a list, and concatenate at the end
            partial_df = self._endpoint_error_handling(predictor, feature_df[index : index + 500])
            df_list.append(partial_df)

        # Concatenate the dataframes
        combined_df = pd.concat(df_list, ignore_index=True)

        # Convert data to numeric
        # Note: Since we're using CSV serializers numeric columns often get changed to generic 'object' types

        # Hard Conversion
        # Note: If are string/object columns we want to use 'ignore' here so those columns
        #       won't raise an error (columns maintain current type)
        converted_df = combined_df.apply(pd.to_numeric, errors="ignore")

        # Soft Conversion
        # Convert columns to the best possible dtype that supports the pd.NA missing value.
        converted_df = converted_df.convert_dtypes()

        # Return the Dataframe
        return converted_df

    def _endpoint_error_handling(self, predictor, feature_df):
        """Internal: Method that handles Errors, Retries, and Binary Search for Error Row(s)"""

        # Convert the DataFrame into a CSV buffer
        csv_buffer = StringIO()
        feature_df.to_csv(csv_buffer, index=False)

        # Error Handling if the Endpoint gives back an error
        try:
            # Send the CSV Buffer to the predictor
            results = predictor.predict(csv_buffer.getvalue())

            # Construct a DataFrame from the results
            results_df = pd.DataFrame.from_records(results[1:], columns=results[0])

            # Capture the return columns
            self.endpoint_return_columns = results_df.columns.tolist()

            # Return the results dataframe
            return results_df

        except botocore.exceptions.ClientError as err:
            if err.response["Error"]["Code"] == "ModelError":  # Model Error
                # Report the error
                self.log.critical(f"Endpoint prediction error: {err.response.get('Message')}")
                if self.exit_on_error:
                    sys.exit(1)

                # Base case: DataFrame with 1 Row
                if len(feature_df) == 1:
                    # If we don't have ANY known good results we're kinda screwed
                    if not self.endpoint_return_columns:
                        raise err

                    # Construct an Error DataFrame (one row of NaNs in the return columns)
                    results_df = self._error_df(feature_df, self.endpoint_return_columns)
                    return results_df

                # Recurse on binary splits of the dataframe
                num_rows = len(feature_df)
                split = int(num_rows / 2)
                first_half = self._endpoint_error_handling(predictor, feature_df[0:split])
                second_half = self._endpoint_error_handling(predictor, feature_df[split:num_rows])
                return pd.concat([first_half, second_half], ignore_index=True)

            else:
                print("Unknown Error from Prediction Endpoint")
                raise err

    def _error_df(self, df, all_columns):
        """Internal: Method to construct an Error DataFrame (a Pandas DataFrame with one row of NaNs)"""
        # Create a new dataframe with all NaNs
        error_df = pd.DataFrame(dict(zip(all_columns, [[np.NaN]] * len(self.endpoint_return_columns))))
        # Now set the original values for the incoming dataframe
        for column in df.columns:
            error_df[column] = df[column].values
        return error_df

    def size(self) -> float:
        """Return the size of this data in MegaBytes"""
        return 0.0

    def aws_meta(self) -> dict:
        """Get ALL the AWS metadata for this artifact"""
        return self.endpoint_meta

    def arn(self) -> str:
        """AWS ARN (Amazon Resource Name) for this artifact"""
        return self.endpoint_meta["EndpointArn"]

    def aws_url(self):
        """The AWS URL for looking at/querying this data source"""
        return f"https://{self.aws_region}.console.aws.amazon.com/athena/home"

    def created(self) -> datetime:
        """Return the datetime when this artifact was created"""
        return self.endpoint_meta["CreationTime"]

    def modified(self) -> datetime:
        """Return the datetime when this artifact was last modified"""
        return self.endpoint_meta["LastModifiedTime"]

    def details(self) -> dict:
        """Additional Details about this Endpoint"""
        details = self.summary()
        return details

    def make_ready(self) -> bool:
        """This is a BLOCKING method that will wait until the Endpoint is ready"""
        self.details()
        self.set_status("ready")
        self.refresh_meta()
        return True

    @staticmethod
    def regression_metrics(target: str, prediction_df: pd.DataFrame) -> dict:
        """Compute the performance metrics for this Endpoint"""

        # Compute the metrics
        metrics = {
            "MAE": mean_absolute_error(prediction_df[target], prediction_df["prediction"]),
            "RMSE": sqrt(mean_squared_error(prediction_df[target], prediction_df["prediction"])),
            "R2": r2_score(prediction_df[target], prediction_df["prediction"]),
        }

        # Return the metrics
        return metrics

    @staticmethod
    def classification_metrics(target: str, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """Compute the performance metrics for this Endpoint
        Args:
            target (str): Name of the target column
            prediction_df (pd.DataFrame): DataFrame with the prediction results
        Returns:
            pd.DataFrame: DataFrame with the performance metrics
        """

        # Get a list of unique labels
        labels = prediction_df[target].unique()

        # Calculate scores
        scores = precision_recall_fscore_support(
            prediction_df[target], prediction_df["prediction"], average=None, labels=labels
        )

        # Put the scores into a dataframe
        score_df = pd.DataFrame(
            {target: labels, "precision": scores[0], "recall": scores[1], "fscore": scores[2], "support": scores[3]}
        )
        print(score_df)

    def delete(self):
        """Delete the Endpoint and Endpoint Config"""

        # Delete endpoint (if it already exists)
        try:
            self.sm_client.delete_endpoint(EndpointName=self.endpoint_name)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint {self.endpoint_name} doesn't exist...")
        try:
            self.sm_client.delete_endpoint_config(EndpointConfigName=self.endpoint_name)
        except botocore.exceptions.ClientError:
            self.log.info(f"Endpoint Config {self.endpoint_name} doesn't exist...")


if __name__ == "__main__":
    """Exercise the Endpoint Class"""
    from sageworks.transforms.pandas_transforms.features_to_pandas import (
        FeaturesToPandas,
    )

    # Grab an Endpoint object and pull some information from it
    my_endpoint = Endpoint("abalone-regression-end")

    # Call the various methods

    # Let's do a check/validation of the Endpoint
    assert my_endpoint.exists()

    # Creation/Modification Times
    print(my_endpoint.created())
    print(my_endpoint.modified())

    # Get the tags associated with this Endpoint
    print(f"Tags: {my_endpoint.sageworks_tags()}")

    # Create the FeatureSet to DF Transform
    feature_to_pandas = FeaturesToPandas("abalone_feature_set")

    # Transform the DataSource into a Pandas DataFrame (with max_rows = 100)
    feature_to_pandas.transform(max_rows=100)

    # Grab the output and show it
    feature_df = feature_to_pandas.get_output()
    print(feature_df)

    # Okay now run inference against our Features DataFrame
    my_prediction_df = my_endpoint.predict(feature_df)
    print(my_prediction_df)

    # Compute performance metrics for out test predictions
    target_column = "class_number_of_rings"
    metrics = my_endpoint.regression_metrics(target_column, my_prediction_df)
    for metric, value in metrics.items():
        print(f"{metric}: {value:0.3f}")
