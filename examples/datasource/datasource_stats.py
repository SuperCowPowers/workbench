from workbench.api.data_source import DataSource
import pandas as pd
from pprint import pprint

# Set Pandas output options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Grab a DataSource and pull some of the EDA Stats
my_data = DataSource("test_data")

# Outliers
outliers = my_data.outliers()
print(outliers)

# Correlations
corr_data = my_data.correlations()
corr_df = pd.DataFrame(corr_data)

# Sort both the columns and index
corr_df.sort_index(inplace=True)
corr_df.sort_index(axis=1, inplace=True)
print(corr_df)

# Get the value counts
value_counts = my_data.value_counts()
pprint(value_counts)
