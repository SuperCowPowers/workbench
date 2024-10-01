"""CleanData: Example Class that demonstrates data cleanup for Light DataSources using Pandas"""

# Local imports
from sageworks.core.transforms.data_to_data.light.data_to_data_light import DataToDataLight
from sageworks.utils import pandas_utils


class CleanData(DataToDataLight):
    """CleanData:Class for filtering, sub-setting, and value constraints on Light DataSources

    Common Usage:
        ```python
        clean_data = CleanData(input_data_uuid, output_data_uuid)
        clean_data.set_output_tags(["abalone", "clean", "whatever"])
        clean_data.transform(drop_na="any", drop_outliers=True, drop_duplicates=True)
        ```
    """

    def __init__(self, input_data_uuid: str, output_data_uuid: str):
        """CleanData Initialization"""

        # Call superclass init
        super().__init__(input_data_uuid, output_data_uuid)

    def transform_impl(self, drop_na="any", drop_duplicates=True, drop_outliers=True, **kwargs):
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

        # Drop Duplicates
        if drop_duplicates:
            self.output_df = pandas_utils.drop_duplicates(self.output_df)

        # Drop Outliers
        if drop_outliers:
            self.output_df = pandas_utils.drop_outliers_iqr(self.output_df, scale=2.0)


if __name__ == "__main__":
    """Exercise the CleanData Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = "test_data"
    output_uuid = "test_data_clean"
    data_to_data = CleanData(input_uuid, output_uuid)
    data_to_data.set_output_tags(["test", "clean"])
    data_to_data.transform(drop_na="any")
