# Views
!!! tip inline end "View Examples"
    Examples of Views are in the [Examples](#examples) section at the bottom of this page. Views are a powerful way to filter and agument your DataSources and FeatureSets. Data Analysis and Feature Engineering are a snap using Views.
    
::: sageworks.core.views.view


## Examples
All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)

**Listing Views**

```py title="views.py"
from sageworks.api.data_source import DataSource

# Convert the Data Source to a Feature Set
test_data = DataSource('test_data')
test_data.views()
["display", "training", "computation"]
```

**Getting a Particular View**

```py title="views.py"
from sageworks.api.feature_set import FeatureSet

fs = FeatureSet('test_features')

# Grab the columns for the display view
display_view = fs.view("display")
display_view.columns
['id', 'name', 'height', 'weight', 'salary', ...]

# Pull the dataframe for this view
df = display_view.pull_dataframe()
	id       name     height      weight         salary	...
0   58  Person 58  71.781227  275.088196  162053.140625  
```

**Query (coming soon)**

All SageWorks Views have an 'offline' store that uses AWS Athena, so any query that you can make with Athena is accessible through the FeatureSet API.

```py title="view_query.py"
from sageworks.api.feature_set import FeatureSet

# Grab a FeatureSet
fs = FeatureSet("abalone_features")

# Make some queries using the Athena backend
df = my_features.query("select * from abalone_features where height > .3")
print(df.head())

df = my_features.query("select * from abalone_features where class_number_of_rings < 3")
print(df.head())
```

**Output**

```python
  sex  length  diameter  height  whole_weight  shucked_weight  viscera_weight  shell_weight  class_number_of_rings
0   M   0.705     0.565   0.515         2.210          1.1075          0.4865        0.5120                     10
1   F   0.455     0.355   1.130         0.594          0.3320          0.1160        0.1335                      8

  sex  length  diameter  height  whole_weight  shucked_weight  viscera_weight  shell_weight  class_number_of_rings
0   I   0.075     0.055   0.010         0.002          0.0010          0.0005        0.0015                      1
1   I   0.150     0.100   0.025         0.015          0.0045          0.0040         0.0050                      2
```

!!! note "Not Finding a particular method?"
    The SageWorks API Classes use the 'Core' Classes Internally, so for an extensive listing of all the methods available please take a deep dive into: [SageWorks Core Classes](../core_classes/overview.md)
