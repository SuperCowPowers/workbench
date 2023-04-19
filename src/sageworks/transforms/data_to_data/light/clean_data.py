"""CleanData: Example Class that demonstrates data cleanup for Light DataSources using Pandas"""

# Local imports
from sageworks.transforms.data_to_data.light.data_to_data_light import DataToDataLight
from sageworks.transforms.pandas_transforms import pandas_utils


class CleanData(DataToDataLight):
    """CleanData:Class for filtering, sub-setting, and value constraints on Light DataSources

    Common Usage:
        clean_data = DataToDataLight(input_data_uuid, output_data_uuid)
        clean_data.set_output_tags(["abalone", "clean", "whatever"])
        clean_data.transform(delete_existing=True/False)
    """

    def __init__(self, input_data_uuid: str, output_data_uuid: str):
        """CleanData Initialization"""

        # Call superclass init
        super().__init__(input_data_uuid, output_data_uuid)

    def transform_impl(self, drop_na="any", **kwargs):
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


if __name__ == "__main__":
    """Exercise the CleanData Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_data_clean"
    CleanData(input_uuid, output_uuid).transform(drop_na="any")

    input_uuid = "test_data_json"
    output_uuid = "test_data_json_clean"
    data_to_data = CleanData(input_uuid, output_uuid)
    data_to_data.set_output_tags(["test", "json", "clean"])
    data_to_data.transform(drop_na="any")
