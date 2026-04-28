# Pull Public LogP Data

Script to download, standardize, and merge publicly available LogP (octanol-water partition coefficient) datasets into a single deduplicated CSV.

## Quick Start

```bash
pip install -r requirements.txt
python pull_logp_data.py
```

Output lands in `output/`. Pull a subset of sources with:

```bash
python pull_logp_data.py --sources tdc opera graphormer
```

## Data Sources

### 1. TDC Lipophilicity_AstraZeneca (~4,200 compounds)
- **Property**: logD at pH 7.4
- **Origin**: AstraZeneca experimental data curated from ChEMBL
- **Access**: `pip install PyTDC` -- loaded programmatically
- **License**: MIT
- **Reference**: https://tdcommons.ai/single_pred_tasks/adme/

### 2. DeepChem / MoleculeNet Lipophilicity (~4,200 compounds)
- **Property**: logD at pH 7.4
- **Origin**: Same underlying ChEMBL data as TDC (cross-check; duplicates removed during merge)
- **Access**: `pip install deepchem` -- `dc.molnet.load_lipo()`
- **License**: MIT
- **Reference**: https://moleculenet.org/datasets-1

### 3. PubChem XLogP (~26,000 compounds)
- **Property**: XLogP (computed partition coefficient)
- **Origin**: NCBI PubChem compound database
- **Access**: PUG REST API (batch query by CID ranges)
- **License**: Public domain
- **Reference**: https://pubchem.ncbi.nlm.nih.gov/

### 4. OPERA / PHYSPROP (~14,050 compounds)
- **Property**: Experimental logP
- **Origin**: EPA PHYSPROP database, curated as OPERA training data (originally used for EPA's KOWWIN model)
- **Access**: Downloaded from the OPERA GitHub repository (NIEHS/OPERA)
- **License**: MIT
- **Reference**: https://github.com/NIEHS/OPERA

### 5. GraphormerLogP / GLP (~42,000 compounds)
- **Property**: Experimental logP
- **Origin**: Multi-source curation by Kazan Federal University (CIMM lab)
- **Access**: Downloaded from GitHub (cimm-kzn/GraphormerLogP)
- **License**: MIT
- **Reference**: https://github.com/cimm-kzn/GraphormerLogP

## Output Format

### Per-source files (`output/logp_<source>.csv`)

| Column | Description |
|--------|-------------|
| `smiles` | Original SMILES from the source |
| `canon_smiles` | RDKit canonical SMILES |
| `logp` | LogP or logD value |
| `source` | Source identifier |

### Merged file (`output/logp_all.csv`)

Deduplicated on canonical SMILES. When a compound appears in multiple sources, values are aggregated:

| Column | Description |
|--------|-------------|
| `canon_smiles` | RDKit canonical SMILES (unique key) |
| `logp_mean` | Mean logP across all sources |
| `logp_std` | Standard deviation (0 if single source) |
| `logp_count` | Number of sources reporting this compound |
| `sources` | Pipe-delimited list of source names |
| `logp_values` | Pipe-delimited individual logP values |

## Other Notable Sources (Not Yet Integrated)

These are additional datasets that could be added in the future:

| Source | Compounds | Notes |
|--------|-----------|-------|
| **EPA CompTox Dashboard** | 750K+ | OPERA predictions + experimental subset; bulk download available |
| **DrugBank** | ~14K drugs | Both experimental and ALOGPS predicted; free academic license required |
| **SAMPL6/7 Challenges** | 11+ | Gold-standard potentiometric measurements; very small but high quality |
| **PharmaBench** (mindrank-ai) | 52K entries across 11 ADMET tasks | CC0 license; LLM-curated from ChEMBL |
| **ADMET-AI** (swansonk14) | 41 ADMET datasets | Pre-trained models on all TDC endpoints |
| **OpenADMET** (OMSF) | Growing | ARPA-H funded open-science consortium |
| **lipophilicity-prediction** (jbr-ai-labs) | 17,603 | Merged logP/logD from multiple sources |
| **Martel et al. UHPLC** | 707 | High-quality UHPLC-measured logP |
