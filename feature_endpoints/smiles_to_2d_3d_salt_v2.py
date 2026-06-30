"""Create the SMILES → salt-keeping 2D + 3D-v2 Molecular Descriptors MetaEndpoint.

Salt-aware sibling of ``smiles-to-2d-3d-v2``: same 3D-v2 leg, but the 2D block
keeps salts (for assays like solubility where the salt form matters):

    [smiles-to-2d-salt-v1]  (sync, RDKit + Mordred 2D, salts kept)  ──┐
                                                                      ├── Concat
    [smiles-to-3d-v2]       (async, curated xTB 3D, 26 features)     ──┘

Note: the 3D leg desalts regardless (conformers/xTB run on the largest fragment),
so the 3D-v2 columns are identical to the non-salt meta — only the 2D block differs.

Because ``smiles-to-3d-v2`` is async (60-minute invocation budget for the GFN2-xTB
conformer work), this MetaEndpoint is auto-deployed as async too — ``MetaEndpoint
.create()`` detects the async child via ``dag.has_async_endpoint()``.

Created artifacts:  Model/Endpoint ``smiles-to-2d-3d-salt-v2``
"""

from workbench.api import MetaEndpoint, PublicData
from workbench.utils.aggregation_nodes import Concat
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG

ENDPOINT_NAME = "smiles-to-2d-3d-salt-v2"
TAGS = ["smiles", "2d", "salt", "3d", "v2", "meta"]

# ─── Autoscaler knobs (async deployment only) ───────────────────────────────
MIN_INSTANCES = 0
MAX_INSTANCES = 1


if __name__ == "__main__":
    # Build the DAG: salt-keeping 2D + 3D-v2 → Concat
    dag = MetaEndpointDAG()
    dag.add_endpoint("smiles-to-2d-salt-v1")
    dag.add_endpoint("smiles-to-3d-v2")
    dag.add_aggregation(Concat(name="combine"))
    dag.add_edge("smiles-to-2d-salt-v1", "combine")
    dag.add_edge("smiles-to-3d-v2", "combine")
    dag.set_input_node("smiles-to-2d-salt-v1", "smiles-to-3d-v2")
    dag.set_output_node("combine")

    # Create + deploy
    end = MetaEndpoint.create(
        name=ENDPOINT_NAME,
        dag=dag,
        description="SMILES → salt-keeping RDKit/Mordred 2D + curated xTB 3D-v2 molecular descriptors",
        tags=TAGS,
        min_instances=MIN_INSTANCES,
        max_instances=MAX_INSTANCES,
    )
    end.set_owner("BW")

    # Smoke test with a few public compounds.
    df = PublicData().get("comp_chem/aqsol/aqsol_public_data")
    end.inference(df[:5])
