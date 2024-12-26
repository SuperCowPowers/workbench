# Molecular Descriptors
In this Blog we'll look at the popular AqSol compound solubility dataset, compute Molecular Descriptors (RDKit and Mordred) and take a deep dive on why NaNs, INFs, and parse errors are generated on about 9% of the compounds.

### Data
AqSolDB: A curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets.
<https://www.nature.com/articles/s41597-019-0151-1>

**Download from Harvard DataVerse:**
<https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OVHAW8>

### Python Packages

- [RDKIT](https://github.com/rdkit/rdkit): Open source toolkit for cheminformatics
- [Mordred Community](https://github.com/JacksonBurns/mordred-community): Community maintained version of the mordred molecular descriptor calculator.


### Canonicalization and Tautomerization
We have another blog on [Canonicalization and Tautomerization](canonicalization_and_tautomerization.md) which covers the importance of getting your compounds into a standardized ...


```
df = DataSource("aqsol_data").pull_dataframe()
mol_df = compute_molecular_descriptors(df)

INFO Computing Molecular Descriptors...
INFO Computing RDKit Descriptors...
INFO Found 2 INF/-INF values. Replacing them with NaN.
INFO INF/-INF value 'inf' found in column 'MaxAbsPartialCharge' at index 3047.
INFO INF/-INF value 'inf' found in column 'MaxPartialCharge' at index 3047.


INFO Imputing missing values using 'median' strategy.
WARNING Imputing BCUT2D_MRLOW replacing 884 values with median(0.05)
WARNING Imputing BCUT2D_LOGPHI replacing 884 values with median(2.20)
WARNING Imputing BCUT2D_MWLOW replacing 884 values with median(10.14)
WARNING Imputing MaxAbsPartialCharge replacing 102 values with median(0.44)
WARNING Imputing BCUT2D_CHGHI replacing 884 values with median(2.14)
WARNING Imputing BCUT2D_LOGPLOW replacing 884 values with median(-2.15)
WARNING Imputing MaxPartialCharge replacing 102 values with median(0.28)
WARNING Imputing MinAbsPartialCharge replacing 101 values with median(0.27)
WARNING Imputing MinPartialCharge replacing 101 values with median(-0.41)
WARNING Imputing BCUT2D_MWHI replacing 884 values with median(16.55)
WARNING Imputing BCUT2D_MRHI replacing 884 values with median(5.93)
WARNING Imputing BCUT2D_CHGLO replacing 884 values with median(-2.09)


INFO Computing Mordred Descriptors...
WARNING Failed to parse value 'float division by zero (RotRatio/nBondsO)' in column 'RotRatio' at index 38.
WARNING Failed to parse value 'float division by zero (RotRatio/nBondsO)' in column 'RotRatio' at index 71.
...
WARNING Failed to parse vallue 'float division by zero (RotRatio/nBondsO)' in column 'RotRatio' at index 5579.


INFO Imputing missing values using 'median' strategy.
WARNING Imputing RotRatio replacing 149 values with median(0.17)

```