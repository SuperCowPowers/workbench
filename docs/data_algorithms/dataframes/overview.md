# Pandas Dataframe Algorithms

!!! tip inline end "Pandas Dataframes"
    Pandas dataframes are obviously not going to scale as well as our Spark and SQL Algorithms, but for 'moderate' sized data these algorithms provide some nice functionality.

**Pandas Dataframe Algorithms**

Workbench has a growing set of algorithms and data processing tools for Pandas Dataframes. In general these algorithm will take a dataframe as input and give you back a dataframe with additional columns.

## Proximity & Neighbors

Nearest-neighbor lookups over compounds — in **fingerprint space** (Tanimoto over SMILES/fingerprints) or **feature space** (standardized Euclidean over numeric descriptors). Use it to find analogs, flag activity cliffs, and probe a model's applicability domain.

Two entry points:

```python
# Precomputed — the proximity a trained model already carries (or None)
prox = model.prox("fingerprint")

# Fresh — built over a FeatureSet (the pre-model, anomaly-hunting path)
prox = fs.prox("fingerprint", target="logS")
prox = fs.prox("features", feature_list=["mollogp", "tpsa"], target="logS")
```

`space` is `"fingerprint"` or `"features"`, and `prox.space` reports which one you got. Passing a `target` also enables target-aware analysis (`ActivityLandscape`, `ResidualFeatures`) and adds neighbor target values to the results.

Query neighbors the same way on either backend:

```python
prox.neighbors(compound_id, n_neighbors=5, include_self=False)   # rows already in the set
prox.neighbors_from_query_df(query_df, n_neighbors=5)            # novel rows
```

Fingerprint results carry a `similarity` column (Tanimoto 0–1; threshold with `min_similarity`); feature-space results carry a `distance` column (threshold with `radius`).

## Reference

::: workbench.algorithms.dataframe.proximity
    options:
      show_root_heading: false

::: workbench.algorithms.dataframe.feature_space_proximity
    options:
      show_root_heading: false
      
::: workbench.algorithms.dataframe.fingerprint_proximity
    options:
      show_root_heading: false

::: workbench.algorithms.dataframe.projection_2d
    options:
      show_root_heading: false

## Questions?
<img align="right" src="../../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 


