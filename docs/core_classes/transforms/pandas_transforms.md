# Pandas Transforms

!!! tip inline end "API Classes"
    The [API Classes](../../api_classes/overview.md) will often provide helpful methods that give you a DataFrame (data_source.query() for instance), so always check out the [API Classes](../../api_classes/overview.md) first.

These Transforms will give you the ultimate in customization and flexibility when creating AWS Machine Learning Pipelines. Grab a Pandas DataFrame from a DataSource or FeatureSet process in whatever way for your use case and simply create another Workbench DataSource or FeatureSet from the resulting DataFrame.

**Lots of Options:**

!!! warning inline end "Not for Large Data"
    Pandas Transforms can't handle large datasets (> 4 GigaBytes). For doing transforma on large data see our **Heavy** [Transforms](overview.md)

- S3 --> DF --> DataSource
- DataSource --> DF --> DataSource
- DataSoruce --> DF --> FeatureSet
- Get Creative!


::: workbench.core.transforms.pandas_transforms
