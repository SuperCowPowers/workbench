"""Conformance: the Local classes keep the shape of their AWS counterparts

A script written against a LocalModel should run against a Model. That promise
holds only as long as the two surfaces agree, and nothing else checks it -- the
classes share a metadata base but no method-level ancestor, so a rename or a
changed parameter on one side is invisible to the other until someone hits it.

Two rules:

- **Declared parity**: the methods the local classes promise must exist on both,
  with the same parameter names in the same order. This is the contract.
- **Incidental parity**: any other public method that happens to exist on both
  must also agree on parameters. This catches the quiet case -- a local method
  that drifts from the AWS one it was modeled on.

Neither rule requires local to implement everything AWS does. Local deliberately
carries less (no monitoring, no promotion, no inference store), and adding a
method to an AWS class must not fail this file.
"""

import inspect

import pytest

from workbench.api import DataSource, Endpoint, FeatureSet, Model
from workbench.local import LocalDataSource, LocalEndpoint, LocalFeatureSet, LocalModel

# Local class -> its AWS counterpart
COUNTERPARTS = {
    LocalDataSource: DataSource,
    LocalFeatureSet: FeatureSet,
    LocalModel: Model,
    LocalEndpoint: Endpoint,
}

# The methods each local class promises to match. A script using only these
# against a local artifact runs unchanged against the AWS one.
DECLARED = {
    LocalDataSource: ["pull_dataframe", "query", "column_types", "columns", "num_rows", "num_columns", "to_features"],
    LocalFeatureSet: ["pull_dataframe", "query", "columns", "num_rows", "num_columns", "to_model"],
    LocalModel: [
        "to_endpoint",
        "list_inference_runs",
        "default_inference_run",
        "get_inference_predictions",
        "get_inference_metrics",
    ],
    LocalEndpoint: ["inference"],
}

# Parameters that legitimately differ: AWS-only knobs on a shared method name.
# Keyed by (method, parameter) so a whole method is never waved through.
ALLOWED_EXTRA_AWS_PARAMS = {
    ("to_features", "auto_one_hot"),
    ("inference", "threads"),
    ("inference", "drop_error_rows"),
    ("pull_dataframe", "include_aws_columns"),
}


def public_members(cls) -> list:
    """Public methods and properties reachable on this class, including inherited ones"""
    return [
        name
        for name in dir(cls)
        if not name.startswith("_")
        and (isinstance(inspect.getattr_static(cls, name, None), property) or callable(getattr(cls, name, None)))
    ]


def member_kind(cls, name: str) -> str:
    """How the member is exposed: "property" or "method\" """
    return "property" if isinstance(inspect.getattr_static(cls, name), property) else "method"


def parameters(cls, method_name: str) -> list:
    """The method's parameter names, minus self"""
    signature = inspect.signature(getattr(cls, method_name))
    return [name for name in signature.parameters if name != "self"]


def parity_failures(local_cls, aws_cls, method_name: str) -> list:
    """Ways the two members disagree, as readable strings"""
    # A property on one side and a method on the other is a break: callers write
    # `.columns` against one and `.columns()` against the other
    local_kind = member_kind(local_cls, method_name)
    aws_kind = member_kind(aws_cls, method_name)
    if local_kind != aws_kind:
        return [f"{method_name}: {local_kind} locally, {aws_kind} in AWS"]
    if local_kind == "property":
        return []

    local_params = parameters(local_cls, method_name)
    aws_params = parameters(aws_cls, method_name)

    # kwargs pass-through on either side means the rest is free-form
    if "kwargs" in local_params or "kwargs" in aws_params:
        local_params = [p for p in local_params if p != "kwargs"]
        aws_params = aws_params[: len(local_params)]

    allowed = {p for m, p in ALLOWED_EXTRA_AWS_PARAMS if m == method_name}
    aws_params = [p for p in aws_params if p not in allowed]

    bad = []
    for position, local_param in enumerate(local_params):
        if position >= len(aws_params):
            bad.append(f"{method_name}: local has extra parameter '{local_param}'")
        elif local_param != aws_params[position]:
            bad.append(
                f"{method_name}: parameter {position} is '{local_param}' locally, '{aws_params[position]}' in AWS"
            )
    return bad


@pytest.mark.parametrize("local_cls", COUNTERPARTS)
def test_declared_methods_exist(local_cls):
    """The contract: every promised method is on both classes"""
    aws_cls = COUNTERPARTS[local_cls]
    missing = [
        f"{cls.__name__}.{name}"
        for name in DECLARED[local_cls]
        for cls in (local_cls, aws_cls)
        if not hasattr(cls, name)
    ]
    assert not missing, f"Declared parity methods are missing: {missing}"


@pytest.mark.parametrize("local_cls", COUNTERPARTS)
def test_declared_methods_agree_on_parameters(local_cls):
    """A promised method called the same way must take the same arguments"""
    aws_cls = COUNTERPARTS[local_cls]
    failures = [f for name in DECLARED[local_cls] for f in parity_failures(local_cls, aws_cls, name)]
    assert not failures, f"{local_cls.__name__} drifted from {aws_cls.__name__}:\n" + "\n".join(failures)


@pytest.mark.parametrize("local_cls", COUNTERPARTS)
def test_shared_method_names_agree_on_parameters(local_cls):
    """Any method on both classes must agree, promised or not

    A local method that shares a name with an AWS one is read as the same method.
    If it isn't, it needs a different name.
    """
    aws_cls = COUNTERPARTS[local_cls]
    shared = [name for name in public_members(local_cls) if name in public_members(aws_cls)]
    failures = [f for name in shared for f in parity_failures(local_cls, aws_cls, name)]
    assert not failures, f"{local_cls.__name__} drifted from {aws_cls.__name__}:\n" + "\n".join(failures)
