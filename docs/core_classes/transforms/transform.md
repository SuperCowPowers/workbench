# Transform

!!! tip inline end "API Classes"
    The [API Classes](../../api_classes/overview.md) will use Transforms internally. So model.to_endpoint() uses the ModelToEndpoint() transform. If you need more control over the Transform you can use the [Core Classes](../../core_classes/overview.md) directly.

The Workbench Transform class is a base/abstract class that defines API implemented by all the child classes (DataLoaders, DataSourceToFeatureSet, ModelToEndpoint, etc).

::: workbench.core.transforms.transform
