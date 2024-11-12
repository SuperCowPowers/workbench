# Caching Overview
!!! tip inline end "Caching is Crazy"
    Yes, but it's a necessary evil, AWS often takes multiple seconds to respond and doesn't like to be spammed. So those two things in combination necessitate the use of Cached Classes. 

## Welcome to the SageWorks Cached Classes

These classes provide caching for the for the most used SageWorks classes. They transparently handle all the details around retrieving and caching results from the underlying classes.

- **[CachedMeta](cached_meta.md):** Manages lists of Artifacts (get all models, endpoints, etc).
- **[CachedDataSource](cached_data_source.md):** Manages AWS Data Catalog and Athena
- **[CachedFeatureSet](cached_feature_set.md):** Manages AWS Feature Store and Feature Groups
- **[CachedModel](cached_model.md):** Manages the training and deployment of AWS Model Groups and Packages
- **[CachedEndpoint](cached_endpoint.md):** Manages the deployment and invocations/inference on AWS Endpoints


!!! note "Examples"
    All of the SageWorks Examples are in the Sageworks Repository under the `examples/` directory. For a full code listing of any example please visit our [SageWorks Examples](https://github.com/SuperCowPowers/sageworks/blob/main/examples)
