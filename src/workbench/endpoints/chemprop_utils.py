"""Dual-use ChemProp dataset helpers — shared by training and endpoint inference.

Lives on the endpoint import surface (per the :mod:`workbench.endpoints` contract)
because the serving ``predict_fn`` builds datapoints the same way training does.
Top-level deps are numpy + chemprop, both supplied by the ``pytorch_chem`` base image.
"""

import numpy as np

from chemprop import data


def safe_batch_size(dataset_len: int, batch_size: int) -> int:
    """Compute a batch size that avoids ChemProp's drop_last behavior.

    ChemProp's build_dataloader sets drop_last=True when len(dataset) % batch_size == 1
    to avoid batch norm issues with single-sample batches. For prediction/inference this
    drops a sample, causing misalignment with the source DataFrame. Bumping batch_size
    by 1 in that case makes the last batch 2 samples instead of 1, avoiding both problems.
    """
    if dataset_len % batch_size == 1:
        return batch_size + 1
    return batch_size


def find_smiles_column(columns: list[str]) -> str:
    """Find SMILES column (case-insensitive match for 'smiles')."""
    smiles_col = next((c for c in columns if c.lower() == "smiles"), None)
    if smiles_col is None:
        raise ValueError("Column list must contain a 'smiles' column (case-insensitive)")
    return smiles_col


def create_molecule_datapoints(
    smiles_list: list[str],
    targets: np.ndarray | None = None,
    extra_descriptors: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    lt_mask: np.ndarray | None = None,
) -> tuple[list[data.MoleculeDatapoint], list[int]]:
    """Create ChemProp MoleculeDatapoints from SMILES strings.

    gt_mask/lt_mask are per-(row, target) boolean arrays for bounded-loss training:
    gt_mask[i, j] == True means target j on row i is right-censored (true value >= y).
    lt_mask[i, j] == True means target j on row i is left-censored (true value <= y).

    Returns the datapoints plus the indices of the SMILES RDKit could parse (rows that
    fail to parse are dropped, so callers must align downstream arrays on these indices).
    """
    from rdkit import Chem

    datapoints, valid_indices = [], []
    targets = np.atleast_2d(np.array(targets)).T if targets is not None and np.array(targets).ndim == 1 else targets

    for i, smi in enumerate(smiles_list):
        if Chem.MolFromSmiles(smi) is None:
            continue
        y = targets[i].tolist() if targets is not None else None
        x_d = extra_descriptors[i] if extra_descriptors is not None else None
        kwargs = {"y": y, "x_d": x_d}
        if gt_mask is not None:
            kwargs["gt_mask"] = gt_mask[i]
        if lt_mask is not None:
            kwargs["lt_mask"] = lt_mask[i]
        datapoints.append(data.MoleculeDatapoint.from_smi(smi, **kwargs))
        valid_indices.append(i)

    return datapoints, valid_indices
