# Caching Overview
!!! tip inline end "Caching is Crazy"
    Yes, but it's a necessary evil for Web Interfaces. AWS APIs (boto3, Sagemaker) often takes multiple seconds to respond and will often throttle requests if spammed. So for quicker response and less spamming we're using Cached Classes for any Web Interface work.

## Welcome to the SageWorks Cached Classes

These classes provide caching for the for the most used SageWorks classes. They transparently handle all the details around retrieving and caching results from the underlying classes.

- **[CachedMeta](cached_meta.md):** Manages lists of Artifacts (get all models, endpoints, etc).
- **[CachedDataSource](cached_data_source.md):** Caches the method results for SageWorks DataSource.
- **[CachedFeatureSet](cached_feature_set.md):** Caches the method results for SageWorks FeatureSets.
- **[CachedModel](cached_model.md):** Caches the method results for SageWorks Models.
- **[CachedEndpoint](cached_endpoint.md):** Caches the method results for SageWorks Endpoints.

!!! note "Examples"
    All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)
