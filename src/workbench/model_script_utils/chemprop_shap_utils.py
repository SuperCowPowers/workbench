"""SHAP utilities for ChemProp MPNN models.

Computes feature-type-level SHAP values by selectively ablating atom and bond
feature categories (e.g., atomic_num, degree, bond_type) and measuring the
prediction change via shap.PermutationExplainer.

Based on the official chemprop v2 Shapley value notebook:
https://chemprop.readthedocs.io/en/latest/shapley_value_with_customized_featurizers.html
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from chemprop import data
from chemprop.featurizers.atom import MultiHotAtomFeaturizer
from chemprop.featurizers.bond import MultiHotBondFeaturizer
from chemprop.featurizers.molgraph.molecule import SimpleMoleculeMolGraphFeaturizer


# =============================================================================
# Custom featurizers with selective feature-type ablation
# =============================================================================
class AblationAtomFeaturizer(MultiHotAtomFeaturizer):
    """Atom featurizer with per-feature-type ablation via a boolean mask.

    Each ablation slot corresponds to one feature category (e.g., atomic_num,
    degree). When a slot is False, the corresponding encoding bits are zeroed
    out for all atoms. Feature names and count are derived dynamically from
    the parent class so they stay in sync if chemprop changes defaults.
    """

    # Human-readable names for the one-hot groups in _subfeats, followed by
    # the two scalar features that the base class appends (aromaticity, mass).
    SUBFEAT_NAMES = [
        "atomic_num",
        "degree",
        "formal_charge",
        "chiral_tag",
        "num_Hs",
        "hybridization",
    ]
    EXTRA_NAMES = ["is_aromatic", "mass"]

    def __init__(self, keep_features=None, **kwargs):
        super().__init__(**kwargs)
        self.n_ablation_features = len(self._subfeats) + len(self.EXTRA_NAMES)
        self.feature_names = self.SUBFEAT_NAMES[: len(self._subfeats)] + self.EXTRA_NAMES
        self.keep_features = keep_features if keep_features is not None else [True] * self.n_ablation_features

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
        for feat, choices, keep in zip(feats, self._subfeats, self.keep_features):
            j = choices.get(feat, len(choices))
            if keep:
                x[i + j] = 1
            i += len(choices) + 1
        n_sub = len(self._subfeats)
        if self.keep_features[n_sub]:
            x[i] = int(a.GetIsAromatic())
        if self.keep_features[n_sub + 1]:
            x[i + 1] = 0.01 * a.GetMass()
        return x


class AblationBondFeaturizer(MultiHotBondFeaturizer):
    """Bond featurizer with per-feature-type ablation via a boolean mask."""

    FEATURE_NAMES = ["bond_type", "is_conjugated", "is_in_ring", "stereo"]

    def __init__(self, keep_features=None, **kwargs):
        super().__init__(**kwargs)
        self.n_ablation_features = len(self.FEATURE_NAMES)
        self.feature_names = list(self.FEATURE_NAMES)
        self.keep_features = keep_features if keep_features is not None else [True] * self.n_ablation_features

    def __call__(self, b):
        x = np.zeros(len(self), int)
        if b is None:
            x[0] = 1
            return x
        i = 1
        bt = b.GetBondType()
        bt_idx = self.bond_types.index(bt) if bt in self.bond_types else len(self.bond_types)
        if self.keep_features[0] and bt_idx < len(self.bond_types):
            x[i + bt_idx] = 1
        i += len(self.bond_types) + 1
        if self.keep_features[1]:
            x[i] = int(b.GetIsConjugated())
        if self.keep_features[2]:
            x[i + 1] = int(b.IsInRing())
        i += 2
        stereo_val = int(b.GetStereo())
        stereo_idx = self.stereo.index(stereo_val) if stereo_val in self.stereo else len(self.stereo)
        if self.keep_features[3]:
            x[i + stereo_idx] = 1
        return x


class AblationMolGraphFeaturizer(SimpleMoleculeMolGraphFeaturizer):
    """MolGraph featurizer that delegates ablation to its atom/bond featurizers."""

    def __init__(self, atom_featurizer, bond_featurizer):
        super().__init__(atom_featurizer=atom_featurizer, bond_featurizer=bond_featurizer)


# =============================================================================
# SHAP computation
# =============================================================================
def compute_chemprop_shap(
    model,
    smiles: list[str],
    extra_descriptors: np.ndarray | None = None,
    max_evals: int = 100,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute feature-type-level SHAP values for a chemprop MPNN.

    Uses shap.PermutationExplainer to measure the importance of each atom/bond
    feature category by selectively ablating them.

    Args:
        model: A trained chemprop MPNN model (already in eval mode).
        smiles: List of SMILES strings to explain.
        extra_descriptors: Optional (n_molecules, n_features) array of extra
            descriptors matching the smiles list. Pass None for SMILES-only models.
        max_evals: Max evaluations per molecule for PermutationExplainer.
        sample_size: Max molecules to sample (from the provided list).
        seed: Random seed for reproducible sampling.

    Returns:
        shap_values: (n_samples, n_features) array of SHAP values.
        feature_names: List of feature names corresponding to columns.
        indices: Array of sampled molecule indices.
    """
    import shap

    # Instantiate ablation featurizers using chemprop's default v2 settings
    atom_feat = AblationAtomFeaturizer.v2()
    bond_feat = AblationBondFeaturizer()

    n_atom = atom_feat.n_ablation_features
    n_bond = bond_feat.n_ablation_features
    n_total = n_atom + n_bond
    feature_names = atom_feat.feature_names + bond_feat.feature_names

    model.eval()

    def predict_with_mask(keep_atom, keep_bond, smi, x_d=None):
        """Single prediction with ablated features via direct PyTorch inference."""
        atom_feat.keep_features = keep_atom
        bond_feat.keep_features = keep_bond
        featurizer = AblationMolGraphFeaturizer(atom_feat, bond_feat)
        dp = data.MoleculeDatapoint.from_smi(smi, x_d=x_d)
        dataset = data.MoleculeDataset([dp], featurizer=featurizer)
        loader = data.build_dataloader(dataset, shuffle=False, batch_size=1)
        with torch.inference_mode():
            for batch in loader:
                bmg, V_d, X_d, *_ = batch
                return model(bmg, V_d, X_d).detach().cpu().numpy().flatten()[0]

    class _ModelWrapper:
        """Callable wrapper for shap.PermutationExplainer."""

        def __init__(self, smi, x_d=None):
            self.smi = smi
            self.x_d = x_d

        def __call__(self, X):
            preds = []
            for row in X:
                ka = [bool(v) for v in row[:n_atom]]
                kb = [bool(v) for v in row[n_atom:]]
                preds.append([predict_with_mask(ka, kb, self.smi, self.x_d)])
            return np.array(preds)

    def binary_masker(mask, x):
        masked = deepcopy(x)
        masked[mask == 0] = 0
        return np.array([masked])

    # Sample molecules
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(smiles))
    indices = rng.choice(len(smiles), size=n, replace=False) if n < len(smiles) else np.arange(n)

    all_features_on = np.array([[1] * n_total])
    all_shap = []

    print(f"Computing SHAP values for {n} molecules ({n_total} feature types)...")
    for count, idx in enumerate(indices):
        smi = smiles[idx]
        x_d = extra_descriptors[idx] if extra_descriptors is not None else None

        wrapper = _ModelWrapper(smi, x_d)
        explainer = shap.PermutationExplainer(wrapper, masker=binary_masker)
        explanation = explainer(all_features_on, max_evals=max_evals)
        all_shap.append(explanation.values.flatten())

        if (count + 1) % 50 == 0:
            print(f"  Progress: {count + 1}/{n}")

    return np.array(all_shap), feature_names, indices


def format_shap_results(
    shap_values: np.ndarray,
    feature_names: list[str],
    sample_ids: pd.Series | None = None,
    id_column: str = "id",
    top_n: int = 10,
) -> tuple[list[tuple[str, float]], pd.DataFrame, pd.DataFrame]:
    """Compute importance ranking and build DataFrames for SHAP output files.

    Takes the raw output of compute_chemprop_shap and produces the three
    artifacts that match the XGB/PyTorch SHAP output format:
      - shap_importance: ranked list of (feature_name, mean_abs_shap)
      - shap_values_df:  top_n SHAP values per sample (for shap_values.csv)
      - feature_vals_df: |SHAP| magnitudes for plot coloring (for shap_feature_values.csv)

    Args:
        shap_values: (n_samples, n_features) array from compute_chemprop_shap.
        feature_names: Feature name list from compute_chemprop_shap.
        sample_ids: Optional Series of sample identifiers to include as first column.
        id_column: Name for the id column when sample_ids is provided.
        top_n: Number of top features to include in the detail DataFrames.

    Returns:
        shap_importance: Sorted list of (feature_name, mean_abs_shap) tuples (all features).
        shap_values_df: DataFrame of SHAP values for the top_n features.
        feature_vals_df: DataFrame of |SHAP| magnitudes for beeswarm plot coloring.
    """
    # Rank all features by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_importance = sorted(
        [(feature_names[i], round(float(mean_abs[i]), 6)) for i in range(len(feature_names))],
        key=lambda x: x[1],
        reverse=True,
    )

    # Select top_n for detail DataFrames
    top_features = [f[0] for f in shap_importance[:top_n]]
    top_indices = [feature_names.index(f) for f in top_features]

    shap_values_df = pd.DataFrame(shap_values[:, top_indices], columns=top_features)

    # Use |SHAP| as the "feature value" for beeswarm plot coloring.
    # Unlike tabular models where each sample has a real feature value (e.g., molecular
    # weight=350), chemprop's ablation features are uniform across molecules (all "on").
    # Using |SHAP| gives per-molecule color variation: high-impact molecules are colored
    # differently from low-impact ones, producing a meaningful gradient in the plot.
    feature_vals_df = pd.DataFrame(
        np.abs(shap_values[:, top_indices]),
        columns=top_features,
    )

    # Prepend id column if provided
    if sample_ids is not None:
        shap_values_df.insert(0, id_column, sample_ids.values)
        feature_vals_df.insert(0, id_column, sample_ids.values)

    return shap_importance, shap_values_df, feature_vals_df
