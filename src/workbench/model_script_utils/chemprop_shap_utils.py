"""SHAP utilities for ChemProp MPNN models.

Computes per-bit SHAP values by selectively ablating individual atom and bond
feature bits (e.g., atom=C, atom=N, degree=2, bond=AROMATIC) and measuring the
prediction change via shap.PermutationExplainer.

Only features that are actually used by the sampled molecules are included,
keeping the feature count manageable (typically 20-35 features).

Based on the official chemprop v2 Shapley value notebook:
https://chemprop.readthedocs.io/en/latest/shapley_value_with_customized_featurizers.html
"""

import numpy as np
import pandas as pd
import torch
from rdkit import Chem

from chemprop import data
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer


# =============================================================================
# Human-readable label builders for each feature group
# =============================================================================
def _atom_bit_labels(featurizer: MultiHotAtomFeaturizer) -> list[str]:
    """Build a human-readable label for every bit in the atom featurizer output."""
    GROUP_NAMES = ["atomic_num", "degree", "formal_charge", "chiral_tag", "num_Hs", "hybridization"]
    CHIRAL_LABELS = {0: "none", 1: "CW", 2: "CCW", 3: "other"}
    PT = Chem.GetPeriodicTable()

    labels = []
    for group_name, choices in zip(GROUP_NAMES, featurizer._subfeats):
        for key in choices:
            if group_name == "atomic_num":
                label = f"atom={PT.GetElementSymbol(key)}"
            elif group_name == "hybridization":
                label = f"hybrid={str(key).split('.')[-1]}"
            elif group_name == "chiral_tag":
                label = f"chiral={CHIRAL_LABELS.get(key, str(key))}"
            elif group_name == "formal_charge":
                label = f"charge={key:+d}"
            else:
                label = f"{group_name}={key}"
            labels.append(label)
        labels.append(f"{group_name}=other")  # unknown bucket

    # Two scalar features appended by base class
    labels.append("is_aromatic")
    labels.append("mass")
    return labels


def _bond_bit_labels(featurizer: MultiHotBondFeaturizer) -> list[str]:
    """Build a human-readable label for every bit in the bond featurizer output."""
    BOND_NAMES = {
        Chem.rdchem.BondType.SINGLE: "SINGLE",
        Chem.rdchem.BondType.DOUBLE: "DOUBLE",
        Chem.rdchem.BondType.TRIPLE: "TRIPLE",
        Chem.rdchem.BondType.AROMATIC: "AROMATIC",
    }
    STEREO_NAMES = {0: "NONE", 1: "ANY", 2: "E/Z_Z", 3: "E/Z_E", 4: "CIS", 5: "TRANS"}

    labels = ["bond=null"]  # First bit is the null-bond indicator
    for bt in featurizer.bond_types:
        labels.append(f"bond={BOND_NAMES.get(bt, str(bt))}")
    labels.append("bond=other")  # unknown bucket
    labels.append("is_conjugated")
    labels.append("is_in_ring")
    for s in featurizer.stereo:
        labels.append(f"stereo={STEREO_NAMES.get(s, str(s))}")
    return labels


# =============================================================================
# Custom featurizers with per-bit ablation
# =============================================================================
class BitAblationAtomFeaturizer(MultiHotAtomFeaturizer):
    """Atom featurizer with per-bit ablation via a boolean mask.

    Each position in keep_mask corresponds to one bit in the output vector.
    When keep_mask[i] is False, that bit is zeroed out for all atoms.
    """

    def __init__(self, keep_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.keep_mask = keep_mask if keep_mask is not None else np.ones(len(self), dtype=bool)

    def __call__(self, a):
        x = np.zeros(len(self))
        if a is None:
            return x

        feats = [
            a.GetAtomicNum(),
            a.GetTotalDegree(),
            a.GetFormalCharge(),
            int(a.GetChiralTag()),
            int(a.GetTotalNumHs()),
            a.GetHybridization(),
        ]
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1

        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()

        # Apply per-bit mask
        x[~self.keep_mask] = 0
        return x


class BitAblationBondFeaturizer(MultiHotBondFeaturizer):
    """Bond featurizer with per-bit ablation via a boolean mask."""

    def __init__(self, keep_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.keep_mask = keep_mask if keep_mask is not None else np.ones(len(self), dtype=bool)

    def __call__(self, b):
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            x[~self.keep_mask] = 0
            return x

        i = 1
        bt = b.GetBondType()
        bt_idx = self.bond_types.index(bt) if bt in self.bond_types else len(self.bond_types)
        if bt_idx < len(self.bond_types):
            x[i + bt_idx] = 1
        i += len(self.bond_types) + 1
        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2
        stereo_val = int(b.GetStereo())
        stereo_idx = self.stereo.index(stereo_val) if stereo_val in self.stereo else len(self.stereo)
        x[i + stereo_idx] = 1

        # Apply per-bit mask
        x[~self.keep_mask] = 0
        return x


class AblationMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """MolGraph featurizer that delegates ablation to its atom/bond featurizers."""

    def __init__(self, atom_featurizer, bond_featurizer):
        super().__init__(atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer)


# =============================================================================
# Analyze molecules: detect active bits and compute per-molecule feature fractions
# =============================================================================
def _analyze_molecules(smiles_list: list[str], atom_feat, bond_feat) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize all molecules, find active bits, and compute feature fractions.

    For each molecule, computes the fraction of atoms/bonds that activate each
    feature bit. For example, atom=N gets the fraction of nitrogen atoms, and
    bond=AROMATIC gets the fraction of aromatic bonds. These fractions provide
    meaningful per-molecule values for beeswarm plot coloring.

    Returns:
        atom_active: boolean array of shape (n_atom_bits,)
        bond_active: boolean array of shape (n_bond_bits,)
        feature_fractions: (n_molecules, n_atom_bits + n_bond_bits) array of
            per-molecule feature fractions in [0, 1].
    """
    n_atom = len(atom_feat)
    n_bond = len(bond_feat)
    n_total = n_atom + n_bond
    n_mols = len(smiles_list)

    atom_active = np.zeros(n_atom, dtype=bool)
    bond_active = np.zeros(n_bond, dtype=bool)
    feature_fractions = np.zeros((n_mols, n_total), dtype=np.float32)

    for mol_idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Accumulate atom feature vectors and track active bits
        n_atoms = mol.GetNumAtoms()
        if n_atoms > 0:
            atom_sum = np.zeros(n_atom)
            for atom in mol.GetAtoms():
                vec = atom_feat(atom)
                atom_active |= vec != 0
                atom_sum += np.abs(vec)
            feature_fractions[mol_idx, :n_atom] = atom_sum / n_atoms

        # Accumulate bond feature vectors and track active bits
        n_bonds = mol.GetNumBonds()
        if n_bonds > 0:
            bond_sum = np.zeros(n_bond)
            for bond in mol.GetBonds():
                vec = bond_feat(bond)
                bond_active |= vec != 0
                bond_sum += np.abs(vec.astype(float))
            feature_fractions[mol_idx, n_atom:] = bond_sum / n_bonds

    return atom_active, bond_active, feature_fractions


# =============================================================================
# Batched inference helper
# =============================================================================
def _batched_predict(model, molgraphs, x_d_array=None, batch_size=128):
    """Run model inference on a list of MolGraphs in sub-batches.

    Args:
        model: A trained chemprop MPNN model (already in eval mode).
        molgraphs (list): List of MolGraph objects to predict on.
        x_d_array (np.ndarray | None): Optional (n, d) array of extra descriptors,
            one row per MolGraph. Pass None for SMILES-only models.
        batch_size (int): Max molecules per forward pass to limit GPU memory.

    Returns:
        np.ndarray: Predictions array. Shape is (n, n_targets) for regression or
            (n, n_classes) for single-task classification (task dim squeezed).
            For multi-task classification: (n, n_tasks, n_classes).
    """
    all_preds = []
    for start in range(0, len(molgraphs), batch_size):
        end = min(start + batch_size, len(molgraphs))
        bmg = data.BatchMolGraph(molgraphs[start:end])
        V_d = None
        if x_d_array is not None:
            X_d = torch.tensor(x_d_array[start:end], dtype=torch.float32)
        else:
            X_d = None
        with torch.inference_mode():
            output = model(bmg, V_d, X_d)
        all_preds.append(output.detach().cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)

    # Squeeze single-task classification: (n, 1, n_classes) -> (n, n_classes)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]

    return preds


# =============================================================================
# SHAP computation (batched leave-one-out ablation)
# =============================================================================
def compute_chemprop_shap(
    model,
    smiles: list[str],
    extra_descriptors: np.ndarray | None = None,
    extra_feature_names: list[str] | None = None,
    sample_size: int = 500,
    seed: int = 42,
    is_classifier: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Compute per-bit feature importance for a chemprop MPNN via ablation.

    Uses batched leave-one-out ablation: for each active feature bit, zeros it
    out across all sampled molecules and measures the prediction change vs the
    baseline (all features on). This is equivalent to first-order Shapley values
    for independent binary features and produces nearly identical rankings to
    PermutationExplainer at ~20-30x the speed.

    For hybrid models (SMILES + extra descriptors), each descriptor column is
    also ablated by replacing it with the sample mean across explained molecules.
    Because the model's X_d_transform standardizes raw descriptors via
    (x - mean) / scale, substituting the column mean maps to 0 after the
    transform — the SHAP-standard "feature absent" baseline.

    Args:
        model (object): A trained chemprop MPNN model (already in eval mode).
        smiles (list): List of SMILES strings to explain.
        extra_descriptors (np.ndarray | None): Optional (n_molecules, n_features) array of extra
            descriptors matching the smiles list. Pass None for SMILES-only models.
        extra_feature_names (list[str] | None): Column names for extra_descriptors.
            Required when extra_descriptors is provided so the descriptors appear
            in the returned feature_names with readable labels.
        sample_size (int): Max molecules to sample (from the provided list).
        seed (int): Random seed for reproducible sampling.
        is_classifier (bool): Whether the model is a classifier (needed to distinguish
            multiclass classification from multi-target regression).

    Returns:
        tuple: A tuple of (shap_values, feature_names, indices, feature_fractions).

            - shap_values: (n_samples, n_classes, n_active_features) array for multiclass
              classification, or (n_samples, n_active_features) for regression.
            - feature_names: List of human-readable feature names for active bits,
              followed by any extra descriptor names.
            - indices: Array of sampled molecule indices.
            - feature_fractions: (n_samples, n_active_features) array of per-molecule
              feature fractions (atom/bond fractions in [0, 1], extra descriptors
              min-max normalized to [0, 1] for beeswarm coloring).
    """
    if extra_descriptors is not None and extra_feature_names is None:
        raise ValueError("extra_feature_names must be provided when extra_descriptors is not None")
    # Create featurizers with default v2 settings
    atom_feat = BitAblationAtomFeaturizer.v2()
    bond_feat = BitAblationBondFeaturizer()
    n_atom_bits = len(atom_feat)
    n_bond_bits = len(bond_feat)

    # Build human-readable labels for ALL bits
    all_atom_labels = _atom_bit_labels(atom_feat)
    all_bond_labels = _bond_bit_labels(bond_feat)

    # Sample molecules
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(smiles))
    indices = rng.choice(len(smiles), size=n, replace=False) if n < len(smiles) else np.arange(n)
    sampled_smiles = [smiles[i] for i in indices]

    # Find active bits and compute per-molecule feature fractions
    atom_active, bond_active, all_fractions = _analyze_molecules(sampled_smiles, atom_feat, bond_feat)
    active_mask = np.concatenate([atom_active, bond_active])
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)

    feature_names = [
        (all_atom_labels[i] if i < n_atom_bits else all_bond_labels[i - n_atom_bits]) for i in active_indices
    ]
    print(f"Active features: {n_active} of {n_atom_bits + n_bond_bits} total bits", flush=True)
    print(f"  Atom: {atom_active.sum()} of {n_atom_bits}, Bond: {bond_active.sum()} of {n_bond_bits}", flush=True)

    model.eval()

    # Pre-parse all sampled molecules once (avoids repeated SMILES parsing)
    mols = [Chem.MolFromSmiles(smi) for smi in sampled_smiles]

    # Single featurizer instance reused across all evaluations
    mg_feat = AblationMolGraphFeaturizer(atom_feat, bond_feat)

    # Full-length mask template (all bits on)
    all_on = np.ones(n_atom_bits + n_bond_bits, dtype=bool)

    # Extra descriptors for sampled molecules
    sampled_x_d = None
    if extra_descriptors is not None:
        sampled_x_d = extra_descriptors[indices].astype(np.float32)

    # --- Baseline predictions (all features on) ---
    print(f"  Building baseline MolGraphs ({n} molecules)...", flush=True)
    atom_feat.keep_mask = all_on[:n_atom_bits]
    bond_feat.keep_mask = all_on[n_atom_bits:]
    baseline_graphs = [mg_feat(mol) for mol in mols]
    baseline_preds = _batched_predict(model, baseline_graphs, sampled_x_d)

    # --- Leave-one-out ablation for each active feature ---
    print(f"  Running leave-one-out ablation for {n_active} features...", flush=True)

    # Multiclass only for classifiers — multi-target regression stays 2D
    is_multiclass = is_classifier and baseline_preds.ndim == 2 and baseline_preds.shape[1] > 1
    n_classes = baseline_preds.shape[1] if is_multiclass else 1

    if is_multiclass:
        # (n_samples, n_classes, n_active_features) — matches XGB/PyTorch convention
        ablation_values = np.zeros((n, n_classes, n_active), dtype=np.float32)
    else:
        ablation_values = np.zeros((n, n_active), dtype=np.float32)

    for feat_idx, global_bit in enumerate(active_indices):
        # Build mask with this one bit turned off
        mask = all_on.copy()
        mask[global_bit] = False
        atom_feat.keep_mask = mask[:n_atom_bits]
        bond_feat.keep_mask = mask[n_atom_bits:]

        # Build ablated MolGraphs for all molecules
        ablated_graphs = [mg_feat(mol) for mol in mols]
        ablated_preds = _batched_predict(model, ablated_graphs, sampled_x_d)

        # Importance = baseline - ablated (positive means feature increases prediction)
        if is_multiclass:
            ablation_values[:, :, feat_idx] = baseline_preds - ablated_preds
        else:
            ablation_values[:, feat_idx] = baseline_preds[:, 0] - ablated_preds[:, 0]

        if (feat_idx + 1) % 10 == 0:
            print(f"  Progress: {feat_idx + 1}/{n_active} features", flush=True)

    print(f"  Ablation complete ({n} molecules x {n_active} graph features)", flush=True)

    # Reset featurizer masks
    atom_feat.keep_mask = all_on[:n_atom_bits]
    bond_feat.keep_mask = all_on[n_atom_bits:]

    # Filter fractions to active-only columns to match ablation values
    active_fractions = all_fractions[:, active_indices]

    # --- Extra descriptor ablation (hybrid models) ---
    # Natural extension of the per-bit ablation approach to hybrid descriptors:
    # for atom/bond bits, "ablate" means set the raw bit to 0 — the value the model
    # sees when the feature is absent. For standardized descriptors, the equivalent
    # "absent" value in the raw input space is the *training mean*, because the
    # model's X_d_transform standardizes via (x - train_mean) / train_scale at eval
    # time, so substituting train_mean produces exactly 0 after the transform.
    # This keeps the ablation semantics identical across graph bits and descriptors.
    if extra_descriptors is not None and extra_feature_names is not None:
        n_extra = sampled_x_d.shape[1]
        if len(extra_feature_names) != n_extra:
            raise ValueError(
                f"extra_feature_names length ({len(extra_feature_names)}) must match "
                f"extra_descriptors columns ({n_extra})"
            )

        print(f"  Running ablation for {n_extra} extra descriptors...", flush=True)
        # Pull the exact training mean from the model's X_d_transform buffers.
        # This is the value that maps to 0 post-standardization, matching the
        # "bit set to 0" semantics used for atom/bond features. ScaleTransform
        # stores the mean buffer as shape (1, n_features) — flatten to (n_features,).
        xd_transform = getattr(model, "X_d_transform", None)
        if xd_transform is not None and hasattr(xd_transform, "mean"):
            extra_means = xd_transform.mean.detach().cpu().numpy().astype(np.float32).ravel()
        else:
            # Fallback for models without a stored transform (shouldn't happen
            # for a properly wired hybrid model but keeps the code defensive).
            extra_means = np.nanmean(sampled_x_d, axis=0)

        if extra_means.shape[0] != n_extra:
            raise ValueError(
                f"X_d_transform mean shape {extra_means.shape} does not match "
                f"n_extra={n_extra}"
            )

        if is_multiclass:
            extra_ablation = np.zeros((n, n_classes, n_extra), dtype=np.float32)
        else:
            extra_ablation = np.zeros((n, n_extra), dtype=np.float32)

        for desc_idx in range(n_extra):
            ablated_x_d = sampled_x_d.copy()
            ablated_x_d[:, desc_idx] = extra_means[desc_idx]
            ablated_preds = _batched_predict(model, baseline_graphs, ablated_x_d)
            if is_multiclass:
                extra_ablation[:, :, desc_idx] = baseline_preds - ablated_preds
            else:
                extra_ablation[:, desc_idx] = baseline_preds[:, 0] - ablated_preds[:, 0]

        # Min-max normalize descriptor values for beeswarm coloring (consistent with
        # atom/bond fractions which are already in [0, 1]).
        extra_min = np.nanmin(sampled_x_d, axis=0)
        extra_max = np.nanmax(sampled_x_d, axis=0)
        extra_range = extra_max - extra_min
        extra_range = np.where(extra_range > 1e-9, extra_range, 1.0)
        extra_fractions = (sampled_x_d - extra_min) / extra_range

        # Concatenate graph + descriptor results along the feature axis
        concat_axis = 2 if is_multiclass else 1
        ablation_values = np.concatenate([ablation_values, extra_ablation], axis=concat_axis)
        active_fractions = np.concatenate([active_fractions, extra_fractions], axis=1)
        feature_names = feature_names + list(extra_feature_names)

        print(f"  Hybrid ablation complete ({n_active} graph + {n_extra} descriptor features)", flush=True)

    return ablation_values, feature_names, indices, active_fractions


def format_shap_results(
    shap_values: np.ndarray,
    feature_names: list[str],
    feature_fractions: np.ndarray,
    sample_ids: pd.Series | None = None,
    id_column: str = "id",
    top_n: int = 10,
) -> tuple[list[tuple[str, float]], pd.DataFrame, pd.DataFrame]:
    """Compute importance ranking and build DataFrames for SHAP output files.

    Takes the raw output of compute_chemprop_shap and produces the three
    artifacts that match the XGB/PyTorch SHAP output format:
      - shap_importance: ranked list of (feature_name, mean_abs_shap)
      - shap_values_df:  top_n SHAP values per sample (for shap_values.csv)
      - feature_vals_df: per-molecule feature fractions (for shap_feature_values.csv)

    Args:
        shap_values: (n_samples, n_features) array from compute_chemprop_shap.
        feature_names: Feature name list from compute_chemprop_shap.
        feature_fractions: (n_samples, n_features) array of per-molecule feature
            fractions (e.g., fraction of atoms that are nitrogen). Used for
            beeswarm plot coloring.
        sample_ids: Optional Series of sample identifiers to include as first column.
        id_column: Name for the id column when sample_ids is provided.
        top_n: Number of top features to include in the detail DataFrames.

    Returns:
        shap_importance: Sorted list of (feature_name, mean_abs_shap) tuples (all features).
        shap_values_df: DataFrame of SHAP values for the top_n features.
        feature_vals_df: DataFrame of per-molecule feature fractions for plot coloring.
    """
    # Rank all features by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_importance = sorted(
        [(feature_names[i], round(float(mean_abs[i]), 6)) for i in range(len(feature_names))],
        key=lambda x: x[1],
        reverse=True,
    )

    # Filter out constant-fraction features (e.g., charge=+0, stereo=NONE).
    # If a feature has near-zero variance across molecules it means every
    # molecule has the same value (e.g., every atom is uncharged).  These
    # features cannot differentiate *why* one molecule behaves differently
    # from another, so they add noise to the beeswarm plot.  They are kept
    # in shap_importance for completeness but excluded from the detail CSVs.
    fraction_std = np.std(feature_fractions, axis=0)
    low_var_names = {feature_names[i] for i in range(len(feature_names)) if fraction_std[i] < 0.01}
    if low_var_names:
        print(f"  Filtering {len(low_var_names)} constant-fraction feature(s): {sorted(low_var_names)}")

    # Select top_n variable features for detail DataFrames
    variable_features = [f[0] for f in shap_importance if f[0] not in low_var_names]
    top_features = variable_features[:top_n]
    top_indices = [feature_names.index(f) for f in top_features]

    shap_values_df = pd.DataFrame(shap_values[:, top_indices], columns=top_features)

    # Use per-molecule feature fractions for beeswarm plot coloring.
    # For example, atom=N gets the fraction of nitrogen atoms in each molecule,
    # bond=AROMATIC gets the fraction of aromatic bonds, etc. This gives
    # meaningful color variation: molecules with high nitrogen content are
    # colored differently from low-nitrogen molecules for the atom=N row.
    feature_vals_df = pd.DataFrame(
        feature_fractions[:, top_indices],
        columns=top_features,
    )

    # Prepend id column if provided
    if sample_ids is not None:
        shap_values_df.insert(0, id_column, sample_ids.values)
        feature_vals_df.insert(0, id_column, sample_ids.values)

    return shap_importance, shap_values_df, feature_vals_df
