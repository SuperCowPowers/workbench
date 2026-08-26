# Workbench Public Data

**Maintainer scripts** that build and publish the datasets at
`s3://workbench-public-data`. End users should consume the data via
`PublicData()` — not by running anything in this directory.

```python
from workbench.api import PublicData
pub = PublicData()
pub.list()                                             # discover datasets
pub.get("comp_chem/logp/logp_all")                     # DataFrame
pub.describe("comp_chem/openadmet/pxr/training/main")  # metadata dict
```

Reads are anonymous — the bucket is public-read, so no AWS credentials are
needed to `list()`, `get()`, or `describe()`.

> **Have a public dataset you'd like to see hosted here?** We're happy to add
> it — contact **support@supercowpowers.com** with the source and license info
> and we'll handle the upload.

## Layout

The S3 key mirrors the path under `output/`, so the local tree *is* the bucket
layout. Datasets that ship a train/test split put them in `training/` and
`testing/` subdirs.

```
common/                              generic non-chemistry fixtures
  abalone, wine_dataset, test_data
comp_chem/
  aqsol/                             AqSolDB solubility
    aqsol_public_data
    alignment/                       base, low/medium/high_overlap
  logp/                              logp_all + per-source files
  logd/                              logd_all + per-source files
  logp_logd/                         overlap_00_03, overlap_03_07, overlap_07_10
  openadmet/                         OpenADMET Consortium challenges
    expansionrx/{training,testing}/  all_endpoints + 9 per-endpoint files
    pxr/{training,testing}/
    asap/{training,testing}/         admet, potency
    octant_cyp/                      inhibition, reactivity, mass_spec_response
  reference_compounds/               curated fixtures backing Workbench tests
  synthetic/multi_task/              controlled multi-task experiment set
```

`descriptions.json` is the top-level index, keyed by full S3 path
(`comp_chem/logp/logp_all.csv`). Every published dataset has an entry with a
description, per-column meanings, row count, license, and source references. It
is the authoritative list of what belongs in the bucket — `upload_data.py
--prune` deletes anything remote it does not describe.

## Licensing

Every entry carries a `license` (SPDX identifier, or `public-domain` /
`unspecified`) and a `license_note` explaining where the term comes from. The
note is where nuance lives — for instance which upstream term governs a merged
file.

| License | Datasets | Sources |
|---------|---------:|---------|
| `CC-BY-4.0` | 27 | OpenADMET ExpansionRx, SangsterLogP, the merged LogP/overlap files, UCI fixtures |
| `MIT` | 22 | OPERA/PHYSPROP, GraphormerLogP, AstraZeneca LogD (MoleculeNet mirror), OpenADMET ASAP, Workbench-authored fixtures |
| `Apache-2.0` | 16 | OpenADMET PXR, CYP, Octant CYP |
| `public-domain` | 6 | PubChem BioAssay (US Government work) |
| `CC0-1.0` | 5 | AqSolDB (Harvard Dataverse) |

Merged and derived files take the most restrictive term among their inputs:
`logp_all` is CC-BY-4.0 because SangsterLogP is, even though its other two
sources are MIT, and the `logp_logd/overlap_*` files inherit from it.

A dataset with no stated upstream terms is marked `unspecified` rather than
assumed open, with the source URL in `references` so the claim is traceable.

`output/` is gitignored, so a fresh clone starts empty. Uploading is
incremental, so you only need to pull the datasets you are actually changing.

## Maintainer Workflow

```bash
pip install -r requirements.txt

python pull_logp_data.py            # -> output/comp_chem/logp/
python pull_logd_data.py            # -> output/comp_chem/logd/
python pull_openadmet_data.py       # -> output/comp_chem/openadmet/

# Push to S3 — maintainer-only, requires AWS credentials for the public bucket.
# Dry run by default; --apply actually uploads.
AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply
```

`upload_data.py` uploads changed CSVs (unchanged files are skipped so
`LastModified` only moves on real content changes — ml_pipeline freshness keys
off it) and merges the local `descriptions.json` into the remote one.

Renames and removals need `--prune`, which treats `descriptions.json` as the
authoritative picture of the bucket: remote keys it does not describe are
deleted, and the remote index is replaced rather than merged. Since the index is
committed and keyed by full S3 path, this stays correct no matter what the
gitignored `output/` tree holds — `--prune` also warns about any local CSV with
no entry, which would otherwise upload and then be pruned on the next run.

```bash
AWS_PROFILE=scp_sandbox_admin python upload_data.py --prune           # review the deletes
AWS_PROFILE=scp_sandbox_admin python upload_data.py --prune --apply
```

## Sources

### OpenADMET (`pull_openadmet_data.py`)

All from the [OpenADMET Consortium](https://openadmet.org/) on HuggingFace.

| Challenge | Contents | License |
|-----------|----------|---------|
| **expansionrx** | 9 ADMET endpoints (LogD, KSOL, HLM/MLM CLint, Caco-2 Papp/efflux, MPPB, MBPB, MGMB), 5,326 train / 2,282 test compounds. Published as one wide `all_endpoints` table plus a per-endpoint file filtered to measured rows. | CC-BY-4.0 |
| **pxr** | hPXR induction pEC50/Emax — primary train set plus counter-assay, single-concentration, semi-pure and HT-chem library variants; blinded and phase-1-revealed test sets. | Apache-2.0 |
| **asap** | ASAP Discovery / Polaris antiviral challenge: 5 ADMET endpoints (560 compounds) and SARS-CoV-2 / MERS-CoV Mpro potency (1,328 compounds), split on the source `Set` column. | MIT |
| **octant_cyp** | Octant CYP3A4 inhibition dose-response, CYP3A4/CYP2J2 reactivity, and LC-MS ionization response for 11,353 compounds. Single release, no train/test split. | Apache-2.0 |

### LogP (`pull_logp_data.py`)

Experimental octanol-water partition coefficients — neutral form, single
species, no pH dependence.

| Source | Compounds | License | Notes |
|--------|-----------|---------|-------|
| **OPERA / PHYSPROP** | ~4,200 | MIT | EPA PHYSPROP curation, training data for OPERA/KOWWIN. [github.com/NIEHS/OPERA](https://github.com/NIEHS/OPERA) |
| **GraphormerLogP (GLP)** | ~42,000 | MIT | Multi-source curation by CIMM Kazan (OpenChem, Huuskonen, SAMPL6/7, etc.). [github.com/cimm-kzn/GraphormerLogP](https://github.com/cimm-kzn/GraphormerLogP) |
| **SangsterLogP** | ~26,000 | CC-BY-4.0 | The most rigorously curated source; its values win in `logp_all`. [Cirino et al., Sci Data 2026](https://doi.org/10.1038/s41597-026-07357-2) |

### LogD (`pull_logd_data.py`)

Experimental octanol-water distribution coefficients at pH 7.4 — pH-dependent,
includes ionized forms. For non-ionizable compounds LogP ≈ LogD, but for
acids/bases they can differ by several log units, which is why the two are kept
separate.

| Source | Compounds | License | Notes |
|--------|-----------|---------|-------|
| **AstraZeneca / ChEMBL** | ~4,200 | MIT (MoleculeNet) | AstraZeneca-measured logD@7.4 from ChEMBL. Fetched from the MoleculeNet S3 mirror — single static CSV, no extra deps. Also redistributed by DeepChem and TDC (as `Lipophilicity_AstraZeneca`). [moleculenet.org](https://moleculenet.org/datasets-1) |

## LogP ↔ LogD Overlap

Both pipelines run the same RDKit + ChEMBL standardization
(`workbench.utils.chem_utils.mol_standardize.MolStandardizer`), so the canonical
`smiles` column is directly joinable across the two merged files:

```python
both = pub.get("comp_chem/logp/logp_all").merge(
    pub.get("comp_chem/logd/logd_all")[["smiles", "logd"]], on="smiles"
)
```

`build_logp_logd_overlap.py` bins that join by |LogP − LogD| into the
`comp_chem/logp_logd/overlap_*` files; `pull_logd_data.py` prints the overlap
count against `logp_all.csv` at the end of its run.

## Output Format

### Per-source files (`comp_chem/<assay>/<assay>_<source>.csv`)

| Column | Description |
|--------|-------------|
| `smiles` | Original SMILES from the source |
| `canon_smiles` | RDKit canonical SMILES (post-standardization) |
| `logp` *or* `logd` | Measured value |
| `source` | Source identifier |

### Merged files (`comp_chem/<assay>/<assay>_all.csv`)

Deduplicated on canonical SMILES; multi-source compounds are aggregated.

| Column | Description |
|--------|-------------|
| `id` | Integer row index |
| `smiles` | RDKit canonical SMILES (unique key) |
| `logp` *or* `logd` | Mean across sources |
| `<value>_std` | Standard deviation (0 if single source) |
| `<value>_count` | Number of sources reporting this compound |
| `sources` | Pipe-delimited source names |
| `<value>_values` | Pipe-delimited individual values |

## Sources Considered but Not Integrated

| Source | Reason |
|--------|--------|
| **Octant `*_wells.tsv`** | Raw per-well plate readings behind the CYP inhibition/reactivity summaries; the fitted curves are what models train on |
| **PubChem XLogP** | *Computed* values (XLogP3 algorithm), not experimental — would dilute the experimental-only set |
| **EPA CompTox Dashboard** | Mostly OPERA *predictions*; experimental subset already covered by PHYSPROP |
| **DrugBank** | Main DrugBank Content is CC BY-NC 4.0 — commercial use needs a separate agreement, so it cannot be redistributed here. The CC0 Open Data (vocabulary, structures) is usable but sits behind a login, so it cannot be pulled unattended like every other source |
| **PharmaBench** | LLM-curated from ChEMBL; not yet validated against experimental ground truth |
| **lipophilicity-prediction (jbr-ai-labs)** | Mixes LogP and LogD; would need to be split before integration |
| **Martel et al. UHPLC** | High-quality but small (707 compounds); useful as a held-out test set, not training |
| **SAMPL6/7** | Already absorbed into GraphormerLogP |
