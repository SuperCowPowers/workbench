# Canonicalization and Tautomerization
In this Blog we'll look at the popular AqSol compound solubility dataset, compute Molecular Descriptors (RDKit and Mordred) and take a deep dive on why NaNs, INFs, and parse errors are generated on about 9% of the compounds.

### Data
AqSolDB: A curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets.
<https://www.nature.com/articles/s41597-019-0151-1>

**Download from Harvard DataVerse:**
<https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8>

### Python Packages

- [RDKIT](https://github.com/rdkit/rdkit): Open source toolkit for cheminformatics

## Canonicalization
Canonicalization is the process of converting a chemical structure into a unique, standard representation. It ensures that structurally identical compounds are represented in the same way, regardless of how they were originally drawn or encoded (e.g., in SMILES format).

### How It Works:
- Algorithms (e.g., RDKit’s `MolToSmiles` with `isomericSmiles=True`) reorder atoms, bonds, and stereochemistry to produce a unique canonical SMILES string or another standardized format.
- Includes standardizing:
  - **Atom ordering**.
  - **Bond configurations**.
  - **Stereochemical information**.

### Why It’s Important:
- **Removes Redundancy**: Different representations of the same compound (e.g., mirror images or re-ordered bonds) are treated as identical.
- **Ensures Consistency**: ML models and data pipelines process identical compounds uniformly.
- **Facilitates Comparison**: Allows for direct comparison of compounds in datasets.
- **Prevents Duplication**: Helps deduplicate datasets by grouping identical molecules.

### When It’s Used:
- Preprocessing datasets to unify chemical representations.
- During compound matching or database searches.



## Tautomerization
Tautomerization is the process of identifying and optionally converting a molecule into a specific tautomeric form. Tautomers are isomers of a compound that differ in the positions of protons and double bonds but are in rapid equilibrium under physiological conditions.

### How It Works:
- Algorithms identify tautomerizable groups (e.g., keto-enol, imine-enamine) and can standardize compounds to:
  - A **preferred tautomeric form** (e.g., keto over enol for simplicity).
  - A **representation-invariant form** to collapse tautomers into a single, standardized version.

### Why It’s Important:
- **Improves Feature Consistency**: ML models treat tautomers as a single entity, reducing variability in descriptor calculations.
- **Biological Relevance**: Focuses on biologically relevant forms (e.g., the keto form is often more stable).
- **Avoids Data Noise**: Reduces noise caused by the presence of multiple tautomers for the same compound.
- **Essential for Drug Discovery**: Tautomers may exhibit different bioactivity, so properly standardizing them ensures consistent analysis.

### When It’s Used:
- Preprocessing compounds for QSAR/QSPR studies.
- Normalizing datasets for machine learning pipelines.
- Ensuring compatibility with descriptor calculations and downstream analyses.



## Key Differences
| **Aspect**         | **Canonicalization**                                 | **Tautomerization**                                     |
|---------------------|-----------------------------------------------------|---------------------------------------------------------|
| **Purpose**         | Standardizes the entire molecule representation.    | Handles tautomeric equilibria and normalizes tautomers. |
| **Scope**           | Covers all aspects of molecular representation.     | Focuses on proton/bond shifts within tautomeric groups. |
| **Output**          | Unique, canonical representation of a molecule.     | A specific or invariant tautomeric form of a molecule.  |
| **Focus**           | Atom order, bond types, stereochemistry.            | Functional groups capable of tautomerization.           |
| **Use Case**        | Dataset deduplication, consistency, comparison.     | Biologically/chemically meaningful normalization.       |



## Importance in ML Pipelines
### Canonicalization:
- Ensures a one-to-one mapping between molecules and their descriptors.
- Removes duplicates and inconsistencies.
- Facilitates reproducibility by unifying chemical representations.

### Tautomerization:
- Ensures tautomers are treated consistently across datasets.
- Produces more reliable molecular descriptors by standardizing proton and bond positions.
- Avoids introducing noise due to the coexistence of multiple tautomeric forms.



## Practical Example in Workbench
### Canonicalization:
- **Input**: `C1=CC=CC=C1` (Benzene) and `c1ccccc1` (alternate representation).
- **Output**: `c1ccccc1` (unique canonical SMILES).

### Tautomerization:
- **Input**: Keto-enol tautomerism: `C=O` ↔ `C-OH`.
- **Output**: Standardized form based on stability or biological relevance (e.g., `C=O` for keto).

---

Both processes are crucial preprocessing steps in Workbench to ensure high-quality, noise-free datasets for QSAR/QSPR modeling and other predictive tasks in drug discovery.