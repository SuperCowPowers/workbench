# Compound ETL 
***From Raw SMILES to ML Pipeline Ready Molecules***

In this blog, we'll walk through the essential steps for preparing molecular data for machine learning pipelines. Starting with raw SMILES strings, we'll demonstrate how to clean, standardize, and preprocess molecules, ensuring they are ready for downstream tasks: feature selection, engineering, and modeling.

### Why Compound ETL?
Raw molecular datasets often contain inconsistencies, salts, and redundant entries. A well-structured ETL (Extract, Transform, Load) pipeline ensures the data is clean, standardized, and reproducible, which is crucial for building reliable ML models. We'll cover the following steps:

1. Validating SMILES
2. Removing duplicates
3. Handling stereochemistry
4. Selecting the largest fragment
5. Canonicalizing molecules
6. Tautomerizing molecules


### Data
**AqSolDB:** A curated reference set of aqueous solubility data, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values for 9,982 unique compounds curated from 9 publicly available datasets.  
Source: [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0151-1)  

**Download from Harvard DataVerse:**  
[Harvard DataVerse: AqSolDB](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8)


### Python Packages

- **[RDKit](https://github.com/rdkit/rdkit):** Open-source toolkit for cheminformatics, used for tasks like SMILES validation, fragment selection, and stereochemistry handling.
- **[Mordred Community](https://github.com/JacksonBurns/mordred-community):** A community-maintained molecular descriptor calculator for feature extraction and engineering.



### ETL Steps
Here are the core steps of our Compound ETL pipeline:

#### 1. Check for Invalid SMILES
Validating the SMILES strings ensures that downstream processing doesn’t fail due to malformed data. This step identifies and filters out invalid or problematic entries.

#### 2. Deduplicate
Duplicate molecules can skew analysis and modeling results. Deduplication ensures a clean and minimal dataset.

#### 3. Handle Stereochemistry
Stereochemistry affects molecular properties significantly. This step determines whether to retain or relax stereochemical definitions, depending on the use case.

#### 4. Select Largest Fragment
Many compounds contain salts or counterions. This step extracts the largest fragment with at least one heavy atom and retains any other fragments as metadata.

#### 5. Canonicalize Molecules
Canonicalization ensures that each molecule is represented in a unique and consistent format. This step is critical for reproducibility and efficient comparison.

#### 6. Tautomerize Molecules
Tautomerization standardizes different tautomeric forms of a compound into a single representation, reducing redundancy and improving consistency.



### Canonicalization and Tautomerization
For an in-depth look at why **Canonicalization** and **Tautomerization** are crucial for compound preprocessing, see our blog on [Canonicalization and Tautomerization](canonicalization_and_tautomerization.md). It covers the importance of standardizing molecular representations to ensure robust and reproducible machine learning workflows.


## Conclusion
By following this Compound ETL pipeline, you can transform raw molecular data into a clean, standardized, and ML-ready format. This foundational preprocessing step sets the stage for effective feature engineering, modeling, and analysis.

Stay tuned for the next blog, where we'll dive into feature engineering for chemical compounds!