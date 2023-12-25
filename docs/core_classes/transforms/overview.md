# Transforms

SageWorks currently has a large set of Transforms that go from one Artifact type to another (e.g. DataSource to FeatureSet). The Transforms will often have **light** and **heavy** versions depending on the scale of data that needs to be transformed.

## Transform Types
- Data Loaders
    - Light
    - Heavy
- Data to Features
    - Light
    - Heavy
- Features to Model
- Model to Endpoint

## Pandas Transforms
There's also a large set of Transforms that will use Pandas Dataframes. These are obviously **light** transforms but they do come in quite handle when working with smaller data.

!!! tip "Want to do some custom processing? use Pandas Transforms!"
     At any point you can grab a DataSource or FeatureSet and pull the data as a dataframe. Now run some processing and then save it back to SageWorks

