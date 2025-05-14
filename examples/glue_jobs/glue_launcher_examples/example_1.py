from workbench.api import DataSource

# Grab a test DataSource
ds = DataSource("abalone_data")
df = ds.pull_dataframe()
print(df.head())
