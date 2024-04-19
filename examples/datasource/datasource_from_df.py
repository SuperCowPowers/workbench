from sageworks.utils.test_data_generator import TestDataGenerator
from sageworks.api.data_source import DataSource
from pprint import pprint


# Create a new Data Source from a Pandas DataFrame
gen_data = TestDataGenerator()
df = gen_data.person_data()
test_data = DataSource(df, name="test_data")
pprint(test_data.details())
