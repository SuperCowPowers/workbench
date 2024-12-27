# Compound Explorer 
***A Workbench based Application***

In this blog, we'll walk through the steps we used for taking a 'pile of SMILES' and tranforming them into  compound data for processing and display in the Compound Explorer application. 

### Why Compound Explorer?
The workbench toolkit has a ton of functionality for constructing end-to-end AWS Machine Learning pipelines. We wanted to construct as application that combined components in the toolkit to create an engaging and informative web application.


### Workbench Pipeline

1. **SMILES Processing**:
   - **Validation**: SMILES are syntactically correct.
   - **Fragment Selection**: Retain the largest fragment (with at least one heavy atom) of each molecule.
   - **Canonicalization**: Generate a unique representation for each molecule.
   - **Tautomerization**: Normalize tautomers to standardize inputs.

2. **Feature Space Proximity Models**:
   - Build **KNN-based proximity graphs** for:
     - **Descriptor Features**: Using molecular descriptors (RDKit, Mordred).
     - **Fingerprints**: Using chemical fingerprints for structural similarity (ECFP)

3. **2D Projections**:
   - **LogP vs pKa**: Provide a chemically intuitive 2D Space.
   - Projections (t-SNE, UMAP, etc):
     - **Descriptor Space**
     - **Fingerprint Space**

4. **Interactive Visualization**:
 - **Hover** displays Molecular drawing    
 - **5 Closest Neighbors (2 sets)**:
     - **Blue Lines**: Descriptor-based neighbors.
     - **Green Lines**: Fingerprint-based neighbors.

### Interactivity Highlights:
- **Neighbor Connections**: Clearly differentiate relationships with color-coded edges.
- **Hover Effects**: Enable chemists to interactively explore molecular neighborhoods.
- **Projection Linking**: Allow toggling between Descriptor and Fingerprint spaces.


### Data
**AqSolDB:** A curated reference set of aqueous solubility data, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values for 9,982 unique compounds curated from 9 publicly available datasets.  
Source: [Nature Scientific Data](https://www.nature.com/articles/s41597-019-0151-1)  

**Download from Harvard DataVerse:**  
[Harvard DataVerse: AqSolDB](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8)


### Python Packages

- **[RDKit](https://github.com/rdkit/rdkit):** Open-source toolkit for cheminformatics, used for tasks like SMILES validation, fragment selection, and stereochemistry handling.
- **[Mordred Community](https://github.com/JacksonBurns/mordred-community):** A community-maintained molecular descriptor calculator for feature extraction and engineering.



### Canonicalization and Tautomerization
For an in-depth look at why **Canonicalization** and **Tautomerization** are crucial for compound preprocessing, see our blog on [Canonicalization and Tautomerization](canonicalization_and_tautomerization.md). It covers the importance of standardizing molecular representations to ensure robust and reproducible machine learning workflows.


## Conclusion


Stay tuned for the next blog, where we'll dive into feature engineering for chemical compounds!