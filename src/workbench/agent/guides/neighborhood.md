# Neighborhood

> nearest neighbors of a compound in fingerprint space (Tanimoto similarity)

Find the structurally closest compounds to a given one. This is the tool behind
activity-cliff analysis, explaining a large residual or high uncertainty, and
finding analogs: pull a compound's neighbors, then look at whether near-identical
structures have very different activity.

## Getting the proximity model

Neighbors come from a model's **v1 UQ model**, which carries a fingerprint
proximity backend (v0 is a plain calibrator and has none):

```python
prox = model.uq_model("v1").prox     # a FingerprintProximity
```

Recent models are v1 (they set `uq_version: v1` by default), so this is the
normal path. Pass `"v1"` explicitly — the bare `uq_model()` reads the version
from the bundle and can fall back to v0.

## Neighbors of a compound in the training set

```python
nbrs = prox.neighbors(compound_id, n_neighbors=6, include_self=False)
```

Returns one row per neighbor, with columns:

| column | meaning |
|---|---|
| `prox.id_column` | the query id (the column name varies by model — e.g. `molecule_name`, `id`) |
| `neighbor_id` | the neighboring compound |
| `similarity` | Tanimoto similarity, 0–1 (higher = more similar) |
| `<target>` | the neighbor's target value (may be `NaN` for a multi-task model's off-task rows) |

Two gotchas:

- **`include_self` defaults to `True`.** A compound is its own closest neighbor
  (similarity 1.0), so pass `include_self=False` to get real neighbors.
- **`n_neighbors` counts before self is dropped.** With `include_self=False`,
  request `k + 1` to get `k` neighbors back.

Don't hardcode the query id column — it's `prox.id_column`. `neighbor_id` and
`similarity` are stable.

## Neighbors of a novel compound

For a compound **not** in the training set, look it up by structure instead:

```python
query_df = df[["smiles"]]                          # a 'smiles' column
nbrs = prox.neighbors_from_query_df(query_df, n_neighbors=5)
```

Result columns are `query_id, neighbor_id, similarity, <target>`. Unparseable
SMILES are dropped with a warning rather than raising.

## Controlling the search

```python
prox.neighbors(cid, min_similarity=0.5, include_self=False)   # all neighbors >= 0.5 Tanimoto
```

`min_similarity` (a Tanimoto floor) overrides `n_neighbors` — use it when "how
many are actually similar" matters more than a fixed count.

## Putting it together

Neighbors + the structure grid (`plotting` guide) is the activity-cliff view:
pull a high-residual compound's neighbors, then render the query and its
neighbors side by side, captioned with activity. Near-identical scaffolds at
opposite ends of the potency scale is the cliff. SMILES/id column conventions are
in the `compounds` guide.
