# Tox21 Dataset Overview

The Tox21 dataset is a public resource developed through the "Toxicology in the 21st Century" initiative. It provides qualitative toxicity measurements for a variety of compounds across multiple biological targets, including nuclear receptors and stress response pathways. This dataset is widely used for developing and benchmarking computational models in toxicology.

## Dataset Access

- **Download Links:**
  - [Tox21 Data Browser](https://tox21.gov/data-and-tools/): Access to chemical structures, annotations, and quality control information. [oai_citation_attribution:7‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)
  - [Tox21 Machine Learning Dataset](https://bioinf.jku.at/research/DeepTox/tox21.html): Contains training and test data with dense and sparse features. [oai_citation_attribution:6‡Bioinformatics JKU](https://bioinf.jku.at/research/DeepTox/tox21.html?utm_source=chatgpt.com)

- **Additional Resources:**
  - [EPA CompTox Chemicals Dashboard](https://comptox.epa.gov/dashboard): Provides chemistry, toxicity, and exposure information for a wide range of chemicals. [oai_citation_attribution:5‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)
  - [PubChem](https://pubchem.ncbi.nlm.nih.gov/): Offers access to large-scale screening data, including Tox21 quantitative high-throughput screening (qHTS) data. [oai_citation_attribution:4‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)

## Dataset Columns

The Tox21 dataset includes the following columns:

| Column Name       | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| `Formula`         | Chemical formula of the compound.                                                             |
| `FW`              | Molecular weight (Formula Weight) of the compound.                                            |
| `DSSTox_CID`      | Unique identifier from the Distributed Structure-Searchable Toxicity (DSSTox) database.       |
| `SR-HSE`          | Stress response assay for heat shock element.                                                 |
| `ID`              | Internal identifier for the compound.                                                         |
| `smiles`          | Simplified Molecular Input Line Entry System representation of the compound's structure.      |
| `molecule`        | Molecular structure information (format may vary).                                            |
| `NR-AR`           | Nuclear receptor assay for androgen receptor.                                                 |
| `SR-ARE`          | Stress response assay for antioxidant response element.                                       |
| `NR-Aromatase`    | Nuclear receptor assay for aromatase enzyme.                                                  |
| `NR-ER-LBD`       | Nuclear receptor assay for estrogen receptor ligand-binding domain.                           |
| `NR-AhR`          | Nuclear receptor assay for aryl hydrocarbon receptor.                                         |
| `SR-MMP`          | Stress response assay for mitochondrial membrane potential.                                   |
| `NR-ER`           | Nuclear receptor assay for estrogen receptor.                                                 |
| `NR-PPAR-gamma`   | Nuclear receptor assay for peroxisome proliferator-activated receptor gamma.                  |
| `SR-p53`          | Stress response assay for p53 tumor suppressor protein.                                       |
| `SR-ATAD5`        | Stress response assay for ATAD5 (ATPase family AAA domain-containing protein 5).              |
| `NR-AR-LBD`       | Nuclear receptor assay for androgen receptor ligand-binding domain.                           |

*Note: Descriptions for some columns are based on standard assay targets; specific details may vary.*

## References

- [Tox21 Data and Tools](https://tox21.gov/data-and-tools/) [oai_citation_attribution:3‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)
- [Tox21 Machine Learning Data Set](https://bioinf.jku.at/research/DeepTox/tox21.html) [oai_citation_attribution:2‡Bioinformatics JKU](https://bioinf.jku.at/research/DeepTox/tox21.html?utm_source=chatgpt.com)
- [EPA CompTox Chemicals Dashboard](https://comptox.epa.gov/dashboard) [oai_citation_attribution:1‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) [oai_citation_attribution:0‡Tox21](https://tox21.gov/data-and-tools/?utm_source=chatgpt.com)