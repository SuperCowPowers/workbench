# Model: smiles_to_monroe_model_script
#
# Description: Computes the 720-d Monroe molecular embedding from SMILES strings.
#     Monroe (MIT, https://github.com/blazejba/monroe, arXiv:2608.18982) is a graph
#     transformer pretrained on ~81M PM6 molecules and 1.56M PubChem BioAssay compounds.
#     Each molecule gets one RDKit conformer (ETKDGv3 + MMFF94s, flat 2D layout when
#     embedding fails) and one pass through the frozen encoder.
#
#     Output is one compressed feature column, `monroe`: the 720 values as comma-separated
#     floats (the model templates expand it, as they do the fingerprint endpoint's
#     `fingerprint`). A molecule that cannot be featurized keeps its row with NaN there.
#
import argparse
import os
from io import StringIO
import logging
import pandas as pd
import json
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

# Local imports
from molecular_utils.mol_standardize import standardize
from monroe.model.constants import EDGE_FEAT_LIST_ONE_HOT, NODE_FEAT_LIST_FLOAT, NODE_FEAT_LIST_ONE_HOT
from monroe.model.featurizer import build_single_graph
from monroe.model.grit import GritTransformer

SCRIPT_VERSION = "0.1.0"

# The `encoder` block of the checkpoint's config.json (Monroe commit 57238ed)
ENCODER_CONFIG = {
    "hidden_dim": 720,
    "num_layers": 10,
    "num_heads": 10,
    "emb_dim": 128,
    "walk_len": 16,
    "rbf_dim": 32,
    "dropout": 0.05,
    "use_stereo_edges": True,
    "zero_vn_edge_rbf": False,
}
ENCODER_FILE = "monroe_encoder.pt"
EMBEDDING_COLUMN = "monroe"


def build_encoder() -> GritTransformer:
    """The Monroe encoder, untrained."""
    return GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **ENCODER_CONFIG,
    )


def save_encoder(checkpoint_path, model_dir):
    """Save the encoder's weights from a Monroe checkpoint into the model directory."""
    state = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    encoder_state = {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")}
    build_encoder().load_state_dict(encoder_state)  # A mismatched checkpoint fails here, not at endpoint startup
    torch.save(encoder_state, os.path.join(model_dir, ENCODER_FILE))


def build_graph(smiles):
    """Monroe's graph for one molecule (through InChI, as in pretraining), or None if it can't be built."""
    try:
        inchi = Chem.MolToInchi(Chem.MolFromSmiles(smiles))
        return build_single_graph(inchi=inchi, stereo_augmentation=True, symmetrize=True)
    except Exception:
        return None


def to_pyg(graph):
    """Featurizer graph -> the Data object the encoder expects."""
    return Data(
        x=torch.as_tensor(graph["node_float"], dtype=torch.float32),
        node_codes=torch.as_tensor(graph["node_codes"], dtype=torch.long),
        edge_index=torch.as_tensor(graph["edge_index"], dtype=torch.long),
        edge_codes=torch.as_tensor(graph["edge_codes"], dtype=torch.long),
        pos_in=torch.as_tensor(graph["pos_rdkit"], dtype=torch.float32),
    )


# TRAINING SECTION
#
# This section (__main__) is where SageMaker will execute the training job
# and save the model artifacts to the model directory.
#
if __name__ == "__main__":
    # Script arguments for input/output directories
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    args = parser.parse_args()

    # workbench.training is imported here, not at module scope: the endpoint imports this script too
    from workbench.training.foundation_models import resolve_foundation_checkpoint

    # Nothing is trained: the pretrained encoder goes into the model artifact
    save_encoder(resolve_foundation_checkpoint("monroe"), args.model_dir)


# Model loading and prediction functions
def model_fn(model_dir):
    encoder = build_encoder()
    encoder.load_state_dict(torch.load(os.path.join(model_dir, ENCODER_FILE), weights_only=True))
    return encoder.eval()


def input_fn(input_data, content_type):
    """Parse input data and return a DataFrame."""
    if not input_data:
        raise ValueError("Empty input data is not supported!")

    # Decode bytes to string if necessary
    if isinstance(input_data, bytes):
        input_data = input_data.decode("utf-8")

    if "text/csv" in content_type:
        return pd.read_csv(StringIO(input_data))
    elif "application/json" in content_type:
        return pd.DataFrame(json.loads(input_data))  # Assumes JSON array of records
    else:
        raise ValueError(f"{content_type} not supported!")


def output_fn(output_df, accept_type):
    """Supports both CSV and JSON output formats."""
    if "text/csv" in accept_type:
        return output_df.to_csv(index=False), "text/csv"
    elif "application/json" in accept_type:
        return (
            output_df.to_json(orient="records"),
            "application/json",
        )  # JSON array of records (NaNs -> null)
    else:
        raise RuntimeError(f"{accept_type} accept type is not supported by this script.")


# Prediction function
def predict_fn(df, model):
    logger = logging.getLogger("workbench")
    logger.info(f"smiles_to_monroe_model_script v{SCRIPT_VERSION} — processing {len(df)} molecules")

    # Standardize the molecule (extract salts) first
    df = standardize(df, extract_salts=True)

    # Featurize, then embed the molecules that produced a graph in one batch
    graphs = [build_graph(smiles) if isinstance(smiles, str) else None for smiles in df["smiles"]]
    rows = [i for i, graph in enumerate(graphs) if graph is not None]
    embeddings = [None] * len(df)
    if rows:
        with torch.no_grad():
            vectors = model(Batch.from_data_list([to_pyg(graphs[i]) for i in rows]))[0].numpy()
        for i, vector in zip(rows, vectors):
            embeddings[i] = ",".join(f"{v:.7g}" for v in vector)  # float32 carries ~7 significant digits
    if len(rows) < len(df):
        logger.warning(f"{len(df) - len(rows)} of {len(df)} molecules could not be featurized (NaN embedding)")

    return df.assign(**{EMBEDDING_COLUMN: embeddings})
