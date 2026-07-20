# Compounds

> SMILES columns in DataFrames; rendering molecules with show()

Most Workbench data is chemistry: a row is a compound, with an id, a SMILES
string, and measured or predicted values. So DataFrames coming back from almost
anywhere carry a SMILES column:

```python
fs.pull_dataframe()             # FeatureSet rows
m.get_inference_predictions()   # predictions, with the inputs alongside
end.inference(df)               # DataFrame in, DataFrame out
pub_data.get(...)               # public datasets
```

## Column naming

Anything stored in a DataSource or FeatureSet comes back **lowercase** — that is
Glue/Athena behavior, not a Workbench choice (see `data_and_features`). So
pipeline data has `smiles`, while raw external data often does not: the public
AqSol CSV has `SMILES`, and client files vary.

Don't hardcode either spelling. Find it:

```python
smiles_col = next(c for c in df.columns if c.lower() == "smiles")
```

Same applies to the id column — check rather than assuming `id`.

## Showing a molecule

The REPL has `show()`, which renders a 2D structure image. It takes a **single
SMILES string**, not a DataFrame or a Series:

```python
show(df["smiles"].iloc[0])              # first compound
show("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")    # caffeine
show(smiles, width=800, height=800)     # bigger; default is 500x500
```

To look at several, loop — but keep it to a handful, since each opens an image:

```python
for smiles in df["smiles"].head(5):
    show(smiles)
```

`show()` needs RDKit, so it is only present when the chemistry extras are
installed. If a SMILES string is invalid it logs an error rather than raising —
nothing appearing usually means a bad structure, not a broken call.

For embedding rather than displaying, `workbench.utils.chem_utils.vis` also has
`img_from_smiles()` and `svg_from_smiles()`.

## Useful context

When a user is looking at a compound, the interesting question is usually *why*
a prediction came out the way it did. Pair the structure with the row:

```python
row = df[df["id"] == compound_id].iloc[0]
show(row["smiles"])
print(row[["id", "solubility", "prediction"]])
```

## Descriptors and features

Don't compute molecular descriptors by hand. Feature endpoints do it
consistently for training and inference — see the `feature_endpoints` guide.
The `workbench.utils.chem_utils` package (standardization, tagging, fingerprints,
projections, toxicity) is the layer underneath, operating on RDKit molecules.
