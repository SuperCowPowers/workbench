"""SHAP utilities for ChemProp MPNN models.

Computes per-bit SHAP values by selectively ablating individual atom and bond
feature bits (e.g., atom=C, atom=N, degree=2, bond=AROMATIC) and measuring the
prediction change via shap.PermutationExplainer.

Only features that are actually used by the sampled molecules are included,
keeping the feature count manageable (typically 20-35 features).

Based on the official chemprop v2 Shapley value notebook:
https://chemprop.readthedocs.io/en/latest/shapley_value_with_customized_featurizers.html
"""

from copy import deepcopy

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
# SHAP computation
# =============================================================================
def compute_chemprop_shap(
    model,
    smiles: list[str],
    extra_descriptors: np.ndarray | None = None,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Compute per-bit SHAP values for a chemprop MPNN.

    Uses shap.PermutationExplainer to measure the importance of each individual
    atom/bond feature bit by selectively ablating them. Only bits that are
    actually active in the sampled molecules are included.

    Args:
        model: A trained chemprop MPNN model (already in eval mode).
        smiles: List of SMILES strings to explain.
        extra_descriptors: Optional (n_molecules, n_features) array of extra
            descriptors matching the smiles list. Pass None for SMILES-only models.
        sample_size: Max molecules to sample (from the provided list).
        seed: Random seed for reproducible sampling.

    Returns:
        shap_values: (n_samples, n_active_features) array of SHAP values.
        feature_names: List of human-readable feature names for active bits.
        indices: Array of sampled molecule indices.
        feature_fractions: (n_samples, n_active_features) array of per-molecule
            feature fractions (e.g., fraction of atoms that are nitrogen).
    """
    import shap

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
    print(f"Active features: {n_active} of {n_atom_bits + n_bond_bits} total bits")
    print(f"  Atom: {atom_active.sum()} of {n_atom_bits}, Bond: {bond_active.sum()} of {n_bond_bits}")

    model.eval()

    def predict_with_mask(full_mask, smi, x_d=None):
        """Single prediction with ablated features via direct PyTorch inference."""
        atom_feat.keep_mask = full_mask[:n_atom_bits]
        bond_feat.keep_mask = full_mask[n_atom_bits:]
        featurizer = AblationMolGraphFeaturizer(atom_feat, bond_feat)
        dp = data.MoleculeDatapoint.from_smi(smi, x_d=x_d)
        dataset = data.MoleculeDataset([dp], featurizer=featurizer)
        loader = data.build_dataloader(dataset, shuffle=False, batch_size=1)
        with torch.inference_mode():
            for batch in loader:
                bmg, V_d, X_d, *_ = batch
                return model(bmg, V_d, X_d).detach().cpu().numpy().flatten()[0]

    # Build the full-length mask (all bits on) and create the "active-only" view
    all_on = np.ones(n_atom_bits + n_bond_bits, dtype=bool)

    class _ModelWrapper:
        """Maps active-only mask vectors to full-length masks for the model."""

        def __init__(self, smi, x_d=None):
            self.smi = smi
            self.x_d = x_d

        def __call__(self, X):
            preds = []
            for row in X:
                full_mask = all_on.copy()
                # Only toggle the active bits based on the SHAP mask
                for j, global_idx in enumerate(active_indices):
                    full_mask[global_idx] = bool(row[j])
                preds.append([predict_with_mask(full_mask, self.smi, self.x_d)])
            return np.array(preds)

    def binary_masker(mask, x):
        masked = deepcopy(x)
        masked[mask == 0] = 0
        return np.array([masked])

    all_features_on = np.array([[1] * n_active])
    all_shap = []

    print(f"Computing SHAP values for {n} molecules ({n_active} active features)...")
    for count, idx in enumerate(indices):
        smi = smiles[idx]
        x_d = extra_descriptors[idx] if extra_descriptors is not None else None

        wrapper = _ModelWrapper(smi, x_d)
        explainer = shap.PermutationExplainer(wrapper, masker=binary_masker)
        explanation = explainer(all_features_on)
        all_shap.append(explanation.values.flatten())

        if (count + 1) % 50 == 0:
            print(f"  Progress: {count + 1}/{n}")

    # Filter fractions to active-only columns to match SHAP values
    active_fractions = all_fractions[:, active_indices]
    return np.array(all_shap), feature_names, indices, active_fractions


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
