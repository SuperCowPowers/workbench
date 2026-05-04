"""MetaEndpoint example: a feature-pipeline DAG that fans out to two
feature endpoints in parallel and merges their outputs into a single
wide DataFrame.

Shape:

    [smiles-to-2d-v1] ──┐
                        ├── Concat ── output
    [smiles-to-3d-fast-v1] ──┘

The deployed MetaEndpoint accepts a SMILES DataFrame and returns one row
per input id with the 2D + 3D feature columns merged. If any child were
async (e.g. ``smiles-to-3d-full-v1``), MetaEndpoint.create() would
auto-detect that and deploy this meta endpoint as async too.

Ensemble use case (parallel structure, different aggregation):

    [model-a] ──┐
    [model-b] ──┼── Mean / WeightedMean / ConfidenceWeighted ── output
    [model-c] ──┘

Just swap ``Concat`` for one of the prediction-aggregation nodes from
:mod:`workbench.utils.aggregation_nodes`.
"""

from workbench.api import Endpoint, MetaEndpoint
from workbench.utils.aggregation_nodes import Concat
from workbench.utils.meta_endpoint_dag import MetaEndpointDAG

NAME = "smiles-to-2d-3d-features"
TAGS = ["meta", "features", "smiles", "2d", "3d"]

# Recreate flag — set to True to rebuild the meta endpoint from scratch
recreate = True

# ─── Build the DAG ──────────────────────────────────────────────────────
dag = MetaEndpointDAG()
dag.add_endpoint("smiles-to-2d-v1")
dag.add_endpoint("smiles-to-3d-fast-v1")
dag.add_aggregation(Concat(name="combine"))
dag.add_edge("smiles-to-2d-v1", "combine")
dag.add_edge("smiles-to-3d-fast-v1", "combine")
dag.set_input_node("smiles-to-2d-v1", "smiles-to-3d-fast-v1")
dag.set_output_node("combine")
dag.validate()

# ─── Create + deploy the MetaEndpoint ───────────────────────────────────
if recreate or not Endpoint(NAME).exists():
    end = MetaEndpoint.create(
        name=NAME,
        dag=dag,
        description="Combined 2D RDKit/Mordred + 3D-fast molecular descriptors",
        tags=TAGS,
    )
    end.set_owner("BW")

# ─── Smoke-test the deployed endpoint ───────────────────────────────────
end = MetaEndpoint(NAME)
print(end.summary())
# Sample inference:
#   import pandas as pd
#   df = pd.DataFrame({"id": [1, 2], "smiles": ["CCO", "c1ccccc1"]})
#   result = end.inference(df)
#   print(result.columns)  # id + smiles + 2D features + 3D features
