# Workbench Transforms

A **Transform** converts one Artifact (a stored entity) into another: S3 to DataSource,
DataSource to FeatureSet, FeatureSet to Model, Model to Endpoint. The `Transform` superclass
holds the small API that every subclass shares.

**Stored Entity:** Stored in one or more AWS Services like Data Catalog, Feature Store, Model Registry, etc.

## The API

`Transform(input_name, output_name)` names the input and output artifacts. `transform()` is the
`@final` template method that runs the three stages in order; subclasses fill in the stages
rather than overriding `transform()`.

```python
@abstractmethod
def transform_impl(self, **kwargs):
    """Implement the Transformation from Input to Output"""

def pre_transform(self, **kwargs):
    """Perform any Pre-Transform operations (optional)"""

@abstractmethod
def post_transform(self, **kwargs):
    """Post-Transform ensures that the output Artifact is ready for use"""
```

`input_type()` and `output_type()` report the `TransformInput`/`TransformOutput` enums the
subclass consumes and produces.

## Tags and Provenance

The output artifact is tagged from two places, both read by `get_aws_tags()` when the subclass
creates the artifact:

- `set_output_tags(tags)` — the user-facing tags, stored as `workbench_tags`
- `add_output_meta(meta)` — additional metadata; seeded with `{"workbench_input": input_name}`,
  which is the artifact's provenance

## Setting the Input

There is no `set_input()` on the base class — the input is whatever the subclass consumes, so
each one defines its own. `PandasToFeatures.set_input(df, id_column=..., input_name=...)` takes a
DataFrame plus the artifact it came from, since a DataFrame carries no provenance of its own.

## Package Layout

Subpackages are named `<input>_to_<output>`. Where both exist, `light` is the Pandas
implementation and `heavy` is the Glue/Spark one. `pandas_transforms` holds the
DataFrame-to-artifact transforms that the rest of them build on.
