"""Create the AQSol FeatureSet (first stage of the aqsol pipeline).

DataSources:
    - aqsol_data
FeatureSets:
    - aqsol_features

Produces the `fs:aqsol_features` artifact that the aqsol model scripts consume.
"""

import pandas as pd

from workbench.api import DataSource, PublicData

if __name__ == "__main__":

    # Pull the public AQSol solubility data
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")

    # Add a solubility classification column for the downstream classifier
    bins = [-float("inf"), -5, -4, float("inf")]
    labels = ["low", "medium", "high"]
    df["solubility_class"] = pd.cut(df["Solubility"], bins=bins, labels=labels)

    # Create the aqsol_data DataSource and roll it up into the aqsol_features FeatureSet
    DataSource(df, name="aqsol_data")
    ds = DataSource("aqsol_data")
    fs = ds.to_features("aqsol_features", id_column="id", tags=["aqsol", "public"])
    fs.set_owner("test")
