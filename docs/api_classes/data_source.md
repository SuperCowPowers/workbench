# DataSource

!!! tip inline end "DataSource Examples"
    Examples of using the DataSource class are in the [Examples](#examples) section at the bottom of this page. S3 data, local files, and Pandas dataframes, DataSource can read data from many different sources.

::: sageworks.api.data_source
    options:
      show_root_heading: false


## Examples
All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

**Create a DataSource from an S3 Path or File Path**

``` py title="datasource_from_s3.py"
from sageworks.api.data_source import DataSource

# Create a DataSource from an S3 Path (or a local file)
source_path = "s3://sageworks-public-data/common/abalone.csv"
# source_path = "/full/path/to/local/file.csv"

my_data = DataSource(source_path)
print(my_data.details())
```

**Create a DataSource from a Pandas Dataframe**

``` py title="datasource_from_df.py"
from sageworks.utils.test_data_generator import TestDataGenerator
from sageworks.api.data_source import DataSource

# Create a DataSource from a Pandas DataFrame
gen_data = TestDataGenerator()
df = gen_data.person_data()

test_data = DataSource(df, name="test_data")
print(test_data.details())
```

**Query a DataSource**

All SageWorks DataSources use AWS Athena, so any query that you can make with Athena is accessible through the DataSource API.

```py title="datasource_query.py"
from sageworks.api.data_source import DataSource

# Grab a DataSource
my_data = DataSource("abalone_data")

# Make some queries using the Athena backend
df = my_data.query("select * from abalone_data where height > .3")
print(df.head())

df = my_data.query("select * from abalone_data where class_number_of_rings < 3")
print(df.head())
```

**Output**

```python
  sex  length  diameter  height  whole_weight  shucked_weight  viscera_weight  shell_weight  class_number_of_rings
0   M   0.705     0.565   0.515         2.210          1.1075          0.4865        0.5120                     10
1   F   0.455     0.355   1.130         0.594          0.3320          0.1160        0.1335                      8

  sex  length  diameter  height  whole_weight  shucked_weight  viscera_weight  shell_weight  class_number_of_rings
0   I   0.075     0.055   0.010         0.002          0.0010          0.0005        0.0015                      1
1   I   0.150     0.100   0.025         0.015          0.0045          0.0040        0.0050                      2
```

**Create a FeatureSet from a DataSource**

```py title="datasource_to_featureset.py"
from sageworks.api.data_source import DataSource

# Convert the Data Source to a Feature Set
test_data = DataSource('test_data')
my_features = test_data.to_features()
print(my_features.details())
```

## SageWorks UI
Whenever a DataSource is created SageWorks performs a comprehensive set of Exploratory Data Analysis techniques on your data, pushes the results into AWS, and provides a detailed web visualization of the results.

<figure style="width: 700px;">
<img alt="sageworks_new_light" src="https://github.com/SuperCowPowers/sageworks/assets/4806709/9126bbe7-902e-409e-9caa-570b054b69e6"">
<figcaption>SageWorks Dashboard: DataSources</figcaption>
</figure>


!!! note "Not Finding a particular method?"
    The SageWorks API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)
