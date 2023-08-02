"""SQL based Quartiles: Compute Quartiles for all the numeric columns in a DataSource using SQL"""
import logging
import pandas as pd

# SageWorks Imports
from sageworks.artifacts.data_sources.data_source_abstract import DataSourceAbstract
from sageworks.utils.sageworks_logging import logging_setup

# Setup Logging
logging_setup()
log = logging.getLogger(__name__)


def quartiles(data_source: DataSourceAbstract) -> dict[dict]:
    """Compute Quartiles for all the numeric columns in a DataSource
    Args:
        data_source(DataSource): The DataSource that we're computing quartiles on
    Returns:
        dict(dict): A dictionary of quartiles for each column in the form
             {'col1': {'min': 0, 'q1': 1, 'median': 2, 'q3': 3, 'max': 4},
              'col2': ...}
    """

    # For every column in the table that is numeric, get the quartiles
    data_source.log.info("Computing Quartiles for all numeric columns (this may take a while)...")
    quartile_data = []
    for column, data_type in zip(data_source.column_names(), data_source.column_types()):
        print(column, data_type)
        if data_type in ["bigint", "double", "int", "smallint", "tinyint"]:
            query = (
                f'SELECT MIN("{column}") AS min, '
                f'approx_percentile("{column}", 0.25) AS q1, '
                f'approx_percentile("{column}", 0.5) AS median, '
                f'approx_percentile("{column}", 0.75) AS q3, '
                f'MAX("{column}") AS max FROM {data_source.table_name}'
            )
            result_df = data_source.query(query)
            result_df["column_name"] = column
            quartile_data.append(result_df)
    quartile_dict = pd.concat(quartile_data).set_index("column_name").to_dict(orient="index")

    # Return the quartile data
    return quartile_dict


if __name__ == "__main__":
    """Exercise the SQL Quartiles Functionality"""
    from sageworks.artifacts.data_sources.data_source import DataSource

    # Setup Pandas output options
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.max_columns", 15)
    pd.set_option("display.width", 1000)

    # Retrieve a Data Source
    my_data = DataSource("abalone_data")

    # Verify that the Athena Data Source exists
    assert my_data.check()

    # What's my SageWorks UUID
    print(f"UUID: {my_data.uuid}")

    # Get quartiles for numeric columns
    my_quartiles = quartiles(my_data)
    print("\nQuartiles")
    print(my_quartiles)
