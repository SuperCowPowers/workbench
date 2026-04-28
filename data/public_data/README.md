# Public Lipophilicity Data (LogP / LogD)

**Maintainer scripts** that build the lipophilicity datasets published at
`s3://workbench-public-data`. End users should consume the data via
`PublicData()` — not by running anything in this directory.

```python
from workbench.api import PublicData
pub = PublicData()
pub.list()                                  # discover datasets
pub.get("comp_chem/logp/logp_all")          # DataFrame
pub.describe("comp_chem/logd/logd_all")     # metadata dict
```

Two distinct properties live here. They are kept separate because they are
different physicochemical measurements:

- **LogP** — neutral-form octanol-water partition coefficient (single species,
  no pH dependence)
- **LogD** — pH-dependent octanol-water distribution coefficient that includes
  ionized forms; nearly always reported at pH 7.4

For non-ionizable compounds LogP ≈ LogD, but for acids/bases they can differ
by several log units.

## Maintainer Workflow

```bash
pip install -r requirements.txt

python pull_logp_data.py        # populates output/logp/
python pull_logd_data.py        # populates output/logd/

# Push to S3 — maintainer-only, requires AWS credentials for the public bucket.
# Dry run by default; --apply actually uploads.
AWS_PROFILE=scp_sandbox_admin python upload_data.py --apply
```

> **Have a public dataset you'd like to see hosted here?** We're happy to add
> it — contact **support@supercowpowers.com** with the source and license info
> and we'll handle the upload.

## Layout

```
data/public_data/
├── pull_logp_data.py     # LogP pipeline
├── pull_logd_data.py     # LogD pipeline
├── pull_common.py        # shared standardization / merge helpers
├── alignment_utils.py    # post-merge sanity checks
├── upload_data.py        # push CSVs + descriptions.json to S3
├── descriptions.json     # local copy of the public-bucket index
└── output/
    ├── logp/
    │   ├── logp_all.csv
    │   ├── logp_opera_physprop.csv
    │   └── logp_graphormer_logp.csv
    └── logd/
        ├── logd_all.csv
        └── logd_astrazeneca_chembl.csv
```

`upload_data.py` mirrors `output/<subdir>/<file>.csv` to
`s3://workbench-public-data/comp_chem/<subdir>/<file>.csv` and merges entries
from the local `descriptions.json` into the top-level
`s3://workbench-public-data/descriptions.json` (existing remote entries for
unrelated datasets are preserved).

## LogP Sources

All values are experimental octanol-water partition coefficients.

| Source | Compounds | License | Notes |
|--------|-----------|---------|-------|
| **OPERA / PHYSPROP** | ~4,200 | MIT | EPA PHYSPROP curation, training data for OPERA/KOWWIN. [github.com/NIEHS/OPERA](https://github.com/NIEHS/OPERA) |
| **GraphormerLogP (GLP)** | ~42,000 | MIT | Multi-source curation by CIMM Kazan (OpenChem, Huuskonen, SAMPL6/7, etc.). [github.com/cimm-kzn/GraphormerLogP](https://github.com/cimm-kzn/GraphormerLogP) |

## LogD Sources

All values are experimental octanol-water distribution coefficients at pH 7.4.

| Source | Compounds | License | Notes |
|--------|-----------|---------|-------|
| **AstraZeneca / ChEMBL** | ~4,200 | MIT (MoleculeNet) | AstraZeneca-measured logD@7.4 from ChEMBL. Fetched directly from the MoleculeNet S3 mirror — single static CSV, no extra deps. Same data is also redistributed by DeepChem and Therapeutic Data Commons (as `Lipophilicity_AstraZeneca`). [moleculenet.org](https://moleculenet.org/datasets-1) |

## LogP ↔ LogD Overlap

Both pipelines run the same RDKit + ChEMBL standardization
(`workbench.utils.chem_utils.mol_standardize.MolStandardizer`), so the
canonical `smiles` column is directly joinable across the two merged files:

```python
import pandas as pd
logp = pd.read_csv("output/logp/logp_all.csv")
logd = pd.read_csv("output/logd/logd_all.csv")
both = logp.merge(logd[["smiles", "logd"]], on="smiles")  # rows where both are reported
```

`pull_logd_data.py` prints the overlap count against `logp_all.csv` at the end
of its run.

## Output Format

### Per-source files (`output/<assay>/<assay>_<source>.csv`)

| Column | Description |
|--------|-------------|
| `smiles` | Original SMILES from the source |
| `canon_smiles` | RDKit canonical SMILES (post-standardization) |
| `logp` *or* `logd` | Measured value |
| `source` | Source identifier |

### Merged files (`output/<assay>/<assay>_all.csv`)

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
| **PubChem XLogP** | *Computed* values (XLogP3 algorithm), not experimental — would dilute the experimental-only set |
| **EPA CompTox Dashboard** | Mostly OPERA *predictions*; experimental subset already covered by PHYSPROP |
| **DrugBank** | Mixed experimental/predicted; requires academic license |
| **PharmaBench** | LLM-curated from ChEMBL; not yet validated against experimental ground truth |
| **lipophilicity-prediction (jbr-ai-labs)** | Mixes LogP and LogD; would need to be split before integration |
| **Martel et al. UHPLC** | High-quality but small (707 compounds); useful as a held-out test set, not training |
| **SAMPL6/7** | Already absorbed into GraphormerLogP |
