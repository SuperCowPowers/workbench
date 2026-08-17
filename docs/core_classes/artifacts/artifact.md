# Artifact

!!! tip inline end "API Classes"
    Found a method here you want to use? The [API Classes](../../api_classes/overview.md) have method pass-through so just call the method on the any class that inherits from the **Artifact** Class and voilà it works the same.

The Workbench Artifact class is a base/abstract class that defines API implemented by all the child classes (DataSource, FeatureSet, Model, Endpoint).

`Artifact` is storage-agnostic: naming rules, the abstract method set, and every
helper expressible in terms of `workbench_meta()`/`upsert_workbench_meta()` —
tags, owner, input, status, health. Those two metadata primitives are abstract,
and two classes provide them:

- **`AWSArtifact`** backs metadata with AWS tags and carries the shared AWS
  session and bucket paths. Every AWS artifact class inherits from it.
- **`LocalArtifact`** backs metadata with an on-disk `meta.json` and carries no
  AWS session, bucket, or ARN.

`arn()`, `aws_url()`, and `aws_meta()` are declared on `AWSArtifact` rather than
on the base, since local artifacts have no such thing.

::: workbench.core.artifact

::: workbench.core.artifacts.aws_artifact
