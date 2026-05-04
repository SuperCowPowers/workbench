"""Create the SMILES → 2D + 3D Molecular Descriptors MetaEndpoint.

Combines two existing feature endpoints into a single inference target:

    [smiles-to-2d-v1]   (sync, RDKit + Mordred 2D, ~313 features)  ──┐
                                                                     ├── Concat
    [smiles-to-3d-full-v1] (async, Boltzmann 3D, 74 features)     ──┘

Caller sends a SMILES DataFrame; the deployed MetaEndpoint fans out to
both children, merges their feature columns, and returns a single wide
row per input.

Because ``smiles-to-3d-full-v1`` is async (60-minute invocation budget
for Boltzmann ensemble computation), this MetaEndpoint is auto-deployed
as async too — ``MetaEndpoint.create()`` detects the async child via
``dag.has_async_endpoint()`` and chooses the deploy mode accordingly.

Created artifacts:  Model/Endpoint ``smiles-to-2d-3d-v1``
"""

from workbench.api import MetaEndpoint, PublicData
from workbench.utils.aggregation_nodes import Concat
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG

ENDPOINT_NAME = "smiles-to-2d-3d-v1"
TAGS = ["smiles", "2d", "3d", "meta"]


if __name__ == "__main__":
    # Build the DAG: 2D + 3D-full → Concat
    dag = MetaEndpointDAG()
    dag.add_endpoint("smiles-to-2d-v1")
    dag.add_endpoint("smiles-to-3d-full-v1")
    dag.add_aggregation(Concat(name="combine"))
    dag.add_edge("smiles-to-2d-v1", "combine")
    dag.add_edge("smiles-to-3d-full-v1", "combine")
    dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-full-v1")
    dag.set_output_node("combine")

    # Create + deploy
    end = MetaEndpoint.create(
        name=ENDPOINT_NAME,
        dag=dag,
        description="SMILES → RDKit/Mordred 2D + Boltzmann 3D molecular descriptors",
        tags=TAGS,
    )
    end.set_owner("BW")

    # Smoke test with a few public compounds.
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    end.inference(df[:5])
