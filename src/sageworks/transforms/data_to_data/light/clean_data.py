"""CleanData: Example Class that demonstrates data cleanup for Light DataSources using Pandas"""

# Local imports
from sageworks.transforms.data_to_data.light.data_to_data_light import DataToDataLight
from sageworks.transforms.pandas_transforms import pandas_utils


class CleanData(DataToDataLight):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """CleanData: Class for filtering, sub-setting, and value constraints on Light DataSources uses Pandas"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

    def transform_impl(self, drop_na='any'):
        """Simple Clean Data, will improve later"""

        """
        Notes for later:
        Cleaning data typically involves two phases: Identification and Remediation.

        - Identification
            - Look for NaNs/NULL
            - Make sure all datatypes are correct (df.info()) if not .. fix...
            - Look at distributions/histograms
            - Look for outliers
        - Remediation
            - Drop
            - Fill/Replace
            - Impute the value (using inference/context to fill in a value)
        """

        # Drop Rows that have NaNs in them
        self.output_df = pandas_utils.drop_nans(self.input_df, how=drop_na)


# Simple test of the CleanData functionality
def test():
    """Test the CleanData Class"""
    import pandas as pd
    from sageworks.artifacts.data_sources.athena_source import AthenaSource

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'aqsol_data'
    output_uuid = 'aqsol_data_clean'
    CleanData(input_uuid, output_uuid).transform(drop_na='any')

    # Grab the output and query it for a dataframe
    output = AthenaSource(output_uuid)
    df = output.query(f"select * from {output_uuid} limit 5")

    # Setup Pandas output options
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 1000)

    # Show the dataframe
    print(df)


if __name__ == "__main__":
    test()
