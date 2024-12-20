from workbench.api.feature_set import FeatureSet
import pandas as pd

# Set Pandas output options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Grab a DataSource and pull some of the EDA Stats
my_features = FeatureSet("abalone_features")

# Make some queries using the Athena backend
df = my_features.query("select * from abalone_features where height > .3")
print(df.head())

df = my_features.query("select * from abalone_features where class_number_of_rings < 3")
print(df.head())
