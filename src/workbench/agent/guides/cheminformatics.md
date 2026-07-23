# Cheminformatics

> how Workbench turns SMILES into fingerprints, descriptors, and features — RDKit/Mordred jump-off points

The molecular representations underneath everything else: fingerprints,
descriptors, standardization. This guide is mostly **pointers** — where the real
code lives and what convention it encodes — not a re-implementation. When a lesson
needs the exact behavior, hit the code.

Two neighbors sit on top of this layer: `compounds` handles SMILES columns and
`show()`; `proximity` does Tanimoto similarity and activity cliffs. Come here
for how the primitives are computed.

## Locate the code (install, never a checkout)

The user ran `pip install workbench`; there is no repo. RDKit and Mordred ship as
dependencies. Find everything dynamically:

```python
import workbench, pathlib
CHEM = pathlib.Path(workbench.__file__).parent / "utils/chem_utils"
```

To read the real signature or default, use `code_search` (grep the workbench
tree) and `introspection` (`help()`, `inspect.getsource` on an object in hand).
For **RDKit and Mordred there are no web docs available here** — `help(Chem.FindMolChiralCenters)`
and `dir(module)` in the REPL are the documentation. Reach for them freely.

## How Workbench computes fingerprints

`chem_utils/fingerprints.py` → `compute_morgan_fingerprints(df, radius=2, n_bits=2048)`.
The choices that aren't obvious:

- **Count, not binary.** Each bit holds how many times the substructure appears
  (clamped to 0–255), stored as a comma-separated string. Count fingerprints beat
  binary for ADMET property prediction — the citation is in the module docstring.
- **radius=2** is ECFP4-equivalent; **2048 bits**.
- **Largest fragment first.** Salts/counterions are stripped
  (`rdMolStandardize.LargestFragmentChooser`) before hashing, so the fingerprint
  describes the parent, not the salt.
- **Unparseable SMILES are silently dropped** — the output can have fewer rows
  than the input. If counts don't line up, that's usually why.

## Fingerprints → matrix → 2D

`chem_utils/projections.py`:

- `fingerprints_to_matrix(series)` — auto-detects count-vector (`"0,3,0,..."`) vs
  bitstring (`"1011..."`) format and returns a dense array.
- `project_fingerprints(df, projection="UMAP")` — UMAP (metric `jaccard`) or TSNE,
  adds `x`/`y`. Falls back to TSNE if UMAP isn't installed, and adapts
  perplexity/`n_neighbors` to small datasets.

## Descriptors

`chem_utils/mol_descriptors.py` → `compute_descriptors(df, include_mordred=True, include_stereo=True)`:
RDKit descriptors + ~85 Mordred descriptors (5 ADMET-relevant modules) + custom
stereochemistry features. 3D descriptors and conformer generation (RDKit ETKDG,
xTB energies, Boltzmann weighting) live in `mol_descriptors_3d.py`.

Don't hand-roll descriptors for training data — feature endpoints compute them
consistently for train and inference (`feature_endpoints`). Use these directly
only for ad-hoc analysis.

## Standardization, salts, tagging

What "standardized" means before a fingerprint or descriptor is computed:

- `mol_standardize.py` → `standardize_smiles(smiles)`, `standardize(...)` —
  canonicalization, normalization, fragment/charge handling.
- `salts.py` → `add_salt_features(df)` — flags and classifies counterions.
- `mol_tagging.py` → `tag_molecules(df)`, `filter_by_tags(...)`,
  `get_tag_summary(...)`, `admet_training_set(...)` — composition/structure/
  physchem/liability tags (PAINS and friends via RDKit FilterCatalog).

## Data-quality lenses (short pointers — hit the code for depth)

The "what's actually wrong in my dataset" checks. Each is a jump-off:

- **Missing stereochemistry** — `mol_descriptors.compute_stereochemistry_features(mol)`
  is what Workbench extracts; RDKit `Chem.FindMolChiralCenters(mol, useLegacyImplementation=False)`
  finds centers, and a SMILES with none of `@ / \` carries no stereo. Note that
  count-Morgan on the largest fragment collapses enantiomers and salts — so a pair
  that looks identical in fingerprint space may differ only in stereo, which makes
  an "activity cliff" (`proximity`) an artifact rather than real SAR.
- **Duplicate / unresolved structures** — `misc.feature_resolution_issues(df, features)`
  surfaces rows that collide on features.
- **Units** — `misc.micromolar_to_log(series)` / `log_to_micromolar(...)`,
  `rollup_experimental_data(...)`, `geometric_mean(...)` for combining replicate
  measurements.

For each, read the function to see exactly what it counts before drawing a
conclusion about the data.
