# Molecular Standardization: Canonicalization, Tautomerization, and Salt Handling
In this blog we'll look at why molecular standardization matters for ML pipelines, what Workbench's feature endpoints actually do under the hood, and how the popular AqSol compound solubility dataset illustrates the challenges of working with real-world chemical data.

## Why Standardization Matters
The same molecule can be represented many different ways in SMILES notation. Benzene alone has multiple valid representations: `C1=CC=CC=C1`, `c1ccccc1`, `C1=CC=C(C=C1)` — all describe the same compound. Drug compounds are even worse: they come as salts, mixtures of tautomers, and with inconsistent stereochemistry annotations.

If you feed these raw SMILES into a descriptor computation pipeline, structurally identical compounds produce different feature vectors. Your ML model sees noise where there should be signal. Standardization eliminates this problem.

## Workbench's Standardization Pipeline
Workbench feature endpoints run a four-step standardization pipeline (based on the [ChEMBL structure pipeline](https://doi.org/10.1186/s13321-020-00456-1)) before computing any molecular descriptors:

### Step 1: Cleanup
Removes explicit hydrogens, disconnects metal atoms from organic fragments, and normalizes functional group representations (e.g., different ways of drawing nitro groups or sulfoxides).

### Step 2: Salt/Fragment Handling
Many drug compounds are stored as salt forms (e.g., sodium acetate `[Na+].CC(=O)[O-]`). Workbench provides **two modes** for handling these:

- **`extract_salts=True`** (default): Identifies and keeps the largest organic fragment, removes counterions, and records the removed salt for traceability. Also distinguishes true salts from mixtures — multiple large neutral organic fragments are flagged as mixtures and logged.
- **`extract_salts=False`**: Keeps the full molecule with all fragments intact and preserves ionic charges. Useful when the salt form itself affects the property you're modeling (e.g., solubility, formulation studies).

```python
# Default: removes salts (ChEMBL standard)
df = standardize(df, extract_salts=True)
# Input:  [Na+].CC(=O)[O-]  →  smiles: CC(=O)O, salt: [Na+]

# Keep salts for salt-dependent properties
df = standardize(df, extract_salts=False)
# Input:  [Na+].CC(=O)[O-]  →  smiles: [Na+].CC(=O)[O-], salt: None
```

### Step 3: Charge Neutralization
When salts are extracted, charges on the parent molecule are neutralized (e.g., `CC(=O)[O-]` → `CC(=O)O`). This step is skipped when keeping salts to preserve ionic character.

### Step 4: Tautomer Canonicalization
Tautomers are isomers that differ in proton and double-bond positions but exist in rapid equilibrium. The classic example is the keto-enol pair. Workbench uses RDKit's tautomer enumerator to pick a canonical form, ensuring that the same compound always produces the same descriptors regardless of which tautomeric form appeared in the source data.

```
# 2-hydroxypyridine and 2-pyridone are the same compound
Oc1ccccn1  →  O=c1cccc[nH]1  (canonical tautomer)
```

## Descriptor Computation
After standardization, Workbench computes ~310 molecular descriptors from three sources:

| Source | Count | Description |
|--------|-------|-------------|
| **RDKit** | ~220 | Constitutional, topological, electronic, lipophilicity, pharmacophore, and ADMET-specific descriptors (TPSA, QED, Lipinski) |
| **Mordred** | ~85 | Five ADMET-focused modules: AcidBase, Aromatic, Constitutional, Chi connectivity indices, and CarbonTypes |
| **Stereochemistry** | 10 | Custom features: R/S center counts, E/Z bond counts, stereo complexity, and fraction-defined metrics |

Invalid molecules receive NaN values rather than being dropped, preserving row alignment with the input DataFrame. The `Ipc` descriptor is excluded due to known overflow issues in RDKit.

## Feature Endpoints: Deployed on AWS
These standardization and descriptor computations run inside Workbench **feature endpoints** — SageMaker-hosted transformer models that take raw SMILES and return standardized structures plus computed descriptors. Two variants are available:

- **`smiles-to-taut-md-stereo-v1`**: Standard pipeline with salt extraction (ChEMBL default)
- **`smiles-to-taut-md-stereo-v1-keep-salts`**: Preserves salt forms for salt-sensitive modeling

Both endpoints can be deployed as serverless (cost-efficient for intermittent workloads) or on dedicated instances for higher throughput.

## The AqSol Dataset: A Real-World Example
[AqSolDB](https://www.nature.com/articles/s41597-019-0151-1) is a curated reference set of aqueous solubility values containing 9,982 unique compounds from 9 publicly available datasets ([Harvard DataVerse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8)).

Running this dataset through the full standardization + descriptor pipeline reveals that roughly **9% of compounds** produce NaNs, INFs, or parse errors in one or more descriptors. Common causes include:

- **Invalid or unusual SMILES**: Organometallic compounds, polymers, or SMILES notation errors that RDKit can't parse
- **Descriptor overflow**: Extremely large or complex molecules that cause numerical issues in certain descriptors
- **Mordred edge cases**: Some Mordred modules return error objects rather than numbers for unusual chemical structures

This is why the pipeline uses `errors="coerce"` for Mordred values and returns NaN rather than crashing — downstream ML pipelines can then handle missing values through imputation or row filtering as appropriate.

## Key Differences: Canonicalization vs Tautomerization
| **Aspect** | **Canonicalization** | **Tautomerization** |
|---|---|---|
| **Purpose** | Standardizes the entire molecular representation | Handles proton/bond-shift equilibria |
| **Scope** | Atom ordering, bond types, stereochemistry | Functional groups capable of tautomerization |
| **Output** | Unique, canonical SMILES string | A specific canonical tautomeric form |
| **Use Case** | Deduplication, consistency, comparison | Consistent descriptors across tautomeric forms |

## References

- **ChEMBL Structure Pipeline**: Bento, A.P., et al. *"An open source chemical structure curation pipeline using RDKit."* Journal of Cheminformatics 12, 51 (2020). [DOI: 10.1186/s13321-020-00456-1](https://doi.org/10.1186/s13321-020-00456-1)
- **RDKit Standardization**: Landrum, G. *"Standardization and Validation with the RDKit."* RSC Open Science (2021). [GitHub Notebook](https://github.com/greglandrum/RSC_OpenScience_Standardization_202104)
- **RDKit**: [https://github.com/rdkit/rdkit](https://github.com/rdkit/rdkit)
- **Mordred**: [https://github.com/mordred-descriptor/mordred](https://github.com/mordred-descriptor/mordred)
- **AqSolDB**: Sorkun, M.C., et al. *"AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds."* Scientific Data 6, 143 (2019). [DOI: 10.1038/s41597-019-0151-1](https://doi.org/10.1038/s41597-019-0151-1)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
