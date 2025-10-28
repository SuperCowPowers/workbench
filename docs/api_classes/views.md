# Views
!!! tip inline end "View Examples"
    Examples of using the Views classes to extend the functionality of Workbench Artifacts are in the [Examples](#examples) section at the bottom of this page. 
    
Views are a powerful way to filter and agument your DataSources and FeatureSets. With Views you can subset columns, rows, and even add data to existing Workbench Artifacts. If you want to compute outliers, runs some statistics or engineer some new features, Views are an easy way to change, modify, and add to DataSources and FeatureSets.

    
::: workbench.core.views.view


## Examples
All of the Workbench Examples are in the Workbench Repository under the `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)

**Listing Views**

```py title="views.py"
from workbench.api.data_source import DataSource

# Convert the Data Source to a Feature Set
test_data = DataSource('test_data')
test_data.views()
["display", "training", "computation"]
```

**Getting a Particular View**

```py title="views.py"
from workbench.api.feature_set import FeatureSet

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

**View Queries**

All Workbench Views are stored in AWS Athena, so any query that you can make with Athena is accessible through the View Query API.

```py title="view_query.py"
from workbench.api.feature_set import FeatureSet

# Grab a FeatureSet View
fs = FeatureSet("abalone_features")
d_view = fs.view("display")

# Make some queries using the Athena backend
df = d_view(f"select * from {d_view.table} where height > .3")
print(df.head())

df = t_view.query("select * from abalone_features where class_number_of_rings < 3")
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

!!! note "Classes to construct View"
    The Workbench Classes used to construct viewss are currently in 'Core'. So you can check out the documentation for those classes here: [Workbench View Creators](../core_classes/views/overview.md)
    

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 
