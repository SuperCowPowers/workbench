from workbench.api.feature_set import FeatureSet
import pandas as pd
from pprint import pprint

# Set Pandas output options
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 1000)

# Grab a FeatureSet and pull some of the EDA Stats
my_features = FeatureSet("test_features")

# Grab some of the EDA Stats
corr_data = my_features.correlations()
corr_df = pd.DataFrame(corr_data)

# Sort both the columns and index (for readability)
corr_df.sort_index(inplace=True)
corr_df.sort_index(axis=1, inplace=True)
print(corr_df)

# Get some outliers
outliers = my_features.outliers()
pprint(outliers.head())

# Full set of EDA Stats
eda_stats = my_features.column_stats()
pprint(eda_stats)
