"""Shared helpers for feature-endpoint deploy scripts.

Internal to this directory — not part of the public Workbench API.
"""

from __future__ import annotations

from workbench.api import FeatureSet


def ensure_featureset(name: str = "feature_endpoint_fs"):
    """Ensure a shared FeatureSet (used as the smoke-test / training source
    for feature endpoints) exists. Backed by the public AqSol dataset.

    Feature endpoints consume SMILES and produce computed descriptors — they
    don't "learn" from training data. The FeatureSet just gives workbench
    something to hang the Model / Endpoint artifacts off of during creation.

    Idempotent — returns the existing FeatureSet if it's already there.
    """
    from workbench.api import PublicData
    from workbench.core.transforms.pandas_transforms import PandasToFeatures

    fs = FeatureSet(name)
    if fs.exists():
        return fs

    aqsol = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    aqsol.columns = aqsol.columns.str.lower()
    to_features = PandasToFeatures(name)
    to_features.set_input(aqsol, id_column="id")
    to_features.set_output_tags(["aqsol", "public"])
    to_features.transform()
    fs = FeatureSet(name)
    fs.set_owner("FeatureEndpoint")
    return fs
