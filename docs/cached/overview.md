# Caching Overview
!!! tip inline end "Caching is Crazy"
    Yes, but it's a necessary evil for Web Interfaces. AWS APIs (boto3, Sagemaker) often takes multiple seconds to respond and will often throttle requests if spammed. So for quicker response and less spamming we're using Cached Classes for any Web Interface work.

## Welcome to the Workbench Cached Classes

These classes provide caching for the for the most used Workbench classes. They transparently handle all the details around retrieving and caching results from the underlying classes.

- **[CachedMeta](cached_meta.md):** Manages lists of Artifacts (get all models, endpoints, etc).
- **[CachedDataSource](cached_data_source.md):** Caches the method results for Workbench DataSource.
- **[CachedFeatureSet](cached_feature_set.md):** Caches the method results for Workbench FeatureSets.
- **[CachedModel](cached_model.md):** Caches the method results for Workbench Models.
- **[CachedEndpoint](cached_endpoint.md):** Caches the method results for Workbench Endpoints.

!!! note "Examples"
    All of the Workbench Examples are in the Workbench Repository under the `examples/` directory. For a full code listing of any example please visit our [Workbench Examples](https://github.com/SuperCowPowers/workbench/blob/main/examples)
