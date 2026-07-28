"""Dual-use ChemProp dataset helpers — shared by training and endpoint inference.

Lives on the endpoint import surface (per the :mod:`workbench.endpoints` contract)
because the serving ``predict_fn`` builds datapoints the same way training does.
Top-level deps are numpy + chemprop; ``create_molecule_datapoints`` additionally needs
rdkit at call time (imported inside the function). All three ship in the ``pytorch_chem``
base image.
"""

import numpy as np

from chemprop import data

# Ceiling on rows per forward pass. Serving batches sit well under this and go through as a
# single batch; training-set-sized frames get chunked rather than materializing at once.
MAX_INFERENCE_BATCH = 4096


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


def predict_ensemble(
    models, datapoints, batch_size: int | None = None, num_workers: int = 0, device=None
) -> np.ndarray:
    """Run every ensemble member over ``datapoints`` and return the raw per-member stack.

    The one forward pass for chemprop — serving's ``predict_fn``, the training out-of-fold
    predictions, and the HPO objective all reach the model through here, so a config is
    scored on the same code path it is deployed on.

    Reduction is the caller's: regression averages to a prediction and takes the standard
    deviation, classification averages probabilities before argmax, and the HPO objective
    scores the mean. Returning the stack keeps that policy out of this function.

    Members and batches are placed on ``device``, defaulting to wherever the first member
    already sits — CPU at serving, the accelerator mid-training. fp32 throughout; mixed
    precision is a training-throughput technique, and UQ calibration is fit on the numbers
    this emits.

    Args:
        models: fitted MPNN members, all sharing one architecture.
        datapoints: :func:`create_molecule_datapoints` output — already RDKit-filtered, so
            row *i* of the result corresponds to datapoint *i*.
        batch_size: rows per forward pass, capped at ``MAX_INFERENCE_BATCH`` when None.
        num_workers: dataloader worker processes.
        device: torch device to run on; the first member's own device when None.

    Returns:
        np.ndarray: ``(n_members, n_rows, n_targets)``. Single-target members are reshaped
        to a trailing axis of 1 so callers index ``[:, :, target_idx]`` unconditionally.
    """
    import torch

    device = torch.device(device) if device is not None else next(models[0].parameters()).device

    dataset = data.MoleculeDataset(datapoints)
    # Every member iterates this same dataset, so featurizing each molecule once and reusing
    # the molgraph pays for itself from the second member on.
    dataset.cache = True
    loader = data.build_dataloader(
        dataset,
        batch_size=safe_batch_size(len(dataset), min(batch_size or MAX_INFERENCE_BATCH, len(dataset))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    stack = []
    for model in models:
        model.eval().to(device)
        batch_preds = []
        with torch.inference_mode():
            for batch in loader:
                # TrainingBatch is (bmg, V_d, X_d, targets, weights, lt_mask, gt_mask); a
                # forward pass needs only the first three.
                bmg, V_d, X_d, *_ = batch
                bmg.to(device)
                V_d = None if V_d is None else V_d.to(device)
                X_d = None if X_d is None else X_d.to(device)
                batch_preds.append(model(bmg, V_d, X_d).detach().cpu().numpy())

        preds = np.concatenate(batch_preds, axis=0)
        if preds.ndim == 3 and preds.shape[1] == 1:
            preds = preds.squeeze(axis=1)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        stack.append(preds)

    return np.stack(stack)


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
    if targets is not None:
        targets = np.asarray(targets)
        if targets.ndim == 1:
            targets = np.atleast_2d(targets).T

    for i, smi in enumerate(smiles_list):
        # Guard the RDKit call: MolFromSmiles(None) raises, and blank strings parse to None.
        if not isinstance(smi, str) or not smi.strip() or Chem.MolFromSmiles(smi) is None:
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
