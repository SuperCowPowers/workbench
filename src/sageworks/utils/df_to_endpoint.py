"""Helper Method to call endpoints with a DataFrame as input"""
import sys
import argparse
from io import StringIO

# Third Party
import numpy as np
import pandas as pd
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from sagemaker import Predictor
import botocore

# We need to capture the columns for the returned dataframe
# so that when an error happens we can 'fill in' an error row
result_columns = list()


# Internal Method to construct an Error DataFrame (a Pandas DataFrame with one row of NaNs)
def _error_df(df, all_columns):
    # Create a new dataframe with all NaNs
    error_df = pd.DataFrame(dict(zip(all_columns, [[np.NaN]] * len(result_columns))))
    # Now set the original values for the incoming dataframe
    for column in df.columns:
        error_df[column] = df[column].values
    return error_df


# Internal Method that handles Errors, Retries, and Binary Search for Error Row(s)
def _dataframe_to_endpoint(predictor, df):
    global result_columns

    # Convert the DataFrame into a CSV buffer
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    # Error Handling if the Endpoint gives back an error
    try:
        # Send the CSV Buffer to the predictor
        results = predictor.predict(csv_buffer.getvalue())

        # Construct a DataFrame from the results
        results_df = pd.DataFrame.from_records(results[1:], columns=results[0])

        # Capture the return columns
        result_columns = results_df.columns.tolist()

        # Return the results dataframe
        return results_df

    except botocore.exceptions.ClientError as err:
        if err.response["Error"]["Code"] == "ModelError":  # Model Error
            # Base case: DataFrame with 1 Row
            if len(df) == 1:
                # If we don't have ANY known good results we're kinda screwed
                if not result_columns:
                    raise err

                # Construct an Error DataFrame (one row of NaNs in the return columns)
                results_df = _error_df(df, result_columns)
                return results_df

            # Recurse on binary splits of the dataframe
            num_rows = len(df)
            split = int(num_rows / 2)
            first_half = _dataframe_to_endpoint(predictor, df[0:split])
            second_half = _dataframe_to_endpoint(predictor, df[split:num_rows])
            return pd.concat([first_half, second_half], ignore_index=True)

        else:
            print("Unknown Error from Prediction Endpoint")
            raise err


def df_to_endpoint(endpoint, df, dropna=True):
    df_list = []
    for index in range(0, len(df), 500):
        print("Processing...")

        # Compute partial DataFrames, add them to a list, and concatenate at the end
        partial_df = _dataframe_to_endpoint(endpoint, df[index : index + 500])
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

    # Drop NaNs
    if dropna:
        converted_df.dropna(inplace=True)

    # Return the Dataframe
    return converted_df


def sdf_to_df(sdf_file_path: str) -> pd.DataFrame:
    """Read an SDF file and return the contents as a Pandas DataFrame"""
    print(f"Reading in SDF File: {sdf_file_path}...")
    return PandasTools.LoadSDF(sdf_file_path, smilesName="SMILES")


def endpoint_to_dataframe_tests():
    # Read in our SDF File
    df = sdf_to_df(args.sdfpath)

    # Invoke an Endpoint
    endpoint_name = "smiles-to-rdkit-mordred"
    endpoint = Predictor(endpoint_name, serializer=CSVSerializer(), deserializer=CSVDeserializer())

    # Use the DataFrame helper method
    print(f"Calling Endpoint: {endpoint_name}...")
    endpoint_results = df_to_endpoint(endpoint, df)
    print(endpoint_results.head())

    # Now replace one of the SMILES with a NaN
    df["SMILES"][1] = np.NaN
    df["SMILES"][3] = np.NaN
    print(f"Calling Endpoint with a NaN SMILES: {endpoint_name}...")
    endpoint_results = df_to_endpoint(endpoint, df)
    print(endpoint_results.head())


if __name__ == "__main__":
    # This import is only needed for the tests
    from rdkit.Chem import PandasTools

    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("sdfpath", type=str, help="SDF File Path")
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Run the tests
    endpoint_to_dataframe_tests()
