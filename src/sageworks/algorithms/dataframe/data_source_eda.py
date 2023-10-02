"""DataSourceEDA: Provide basic EDA (Exploratory Data Analysis) for a DataFrame"""
import pandas as pd
import logging

# Local Imports
from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.utils import pandas_utils
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()


class DataSourceEDA:
    def __init__(self, data_source_uuid: str):
        """DataSourceEDA: Provide basic EDA (Exploratory Data Analysis) for a DataSource
        Args:
            data_source_uuid (AthenaSource): DataSource for Exploratory Data Analysis"""
        self.log = logging.getLogger("sageworks")
        self.data_source_uuid = data_source_uuid

        # Spin up the DataToPandas class
        self.data_to_pandas = DataToPandas(self.data_source_uuid)
        self.log.info(f"Getting DataFrame from {self.data_source_uuid}...")
        self.df = self.data_to_pandas.get_output()

    def get_column_info(self):
        """Return the Column Information for the DataSource"""
        column_info_df = pandas_utils.info(self.df)
        return column_info_df

    def get_numeric_stats(self):
        """Return the Column Information for the DataSource"""
        stats_df = pandas_utils.numeric_stats(self.df)
        return stats_df


if __name__ == "__main__":
    """Exercise the DataSourceEDA Class"""

    # Set some pandas options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Spin up the class and get the EDA output
    my_eda = DataSourceEDA("abalone_data")
    print(my_eda.get_column_info())
    print(my_eda.get_numeric_stats())
