from workbench.api.data_source import DataSource
from pprint import pprint

# Create a new Data Source from an S3 Path (or a local file)
source_path = "s3://workbench-public-data/common/abalone.csv"
# source_path = "/full/path/to/local/file.csv"
my_data = DataSource(source_path)
pprint(my_data.summary())
pprint(my_data.details())
