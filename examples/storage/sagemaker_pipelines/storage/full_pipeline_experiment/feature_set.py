from workbench.api.data_source import DataSource

# Create the abalone_data FeatureSet
ds = DataSource("sp_abalone_data")
ds.to_features("sp_abalone_features")
