from workbench.api.data_source import DataSource
from pprint import pprint

# Convert the Data Source to a Feature Set
test_data = DataSource("test_data")
my_features = test_data.to_features("test_features", id_column="id")
pprint(my_features.details())
