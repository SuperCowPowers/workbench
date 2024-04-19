from sageworks.api.data_source import DataSource
import pandas as pd

# Set Pandas output options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Grab a DataSource and pull some of the EDA Stats
my_data = DataSource("abalone_data")

# Make some queries using the Athena backend
df = my_data.query("select * from abalone_data where height > .3")
print(df.head())

df = my_data.query("select * from abalone_data where class_number_of_rings < 3")
print(df.head())
