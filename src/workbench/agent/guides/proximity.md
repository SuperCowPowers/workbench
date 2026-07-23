# Proximity

> nearest neighbors of a compound — fingerprint space (Tanimoto) or feature space (Euclidean)

Find the closest compounds to a given one. This is the tool behind activity-cliff
analysis, explaining a large residual or high uncertainty, and finding analogs:
pull a compound's neighbors, then look at whether near-identical structures have
very different activity.

Two proximity flavors, same interface:

- **FingerprintProximity** — structural similarity from SMILES/fingerprints (Tanimoto).
  Results carry a `similarity` column (0–1); the search floor is `min_similarity`.
- **FeatureSpaceProximity** — closeness in numeric descriptors (standardized Euclidean).
  Results carry a `distance` column; the search bound is `radius`.

## Getting a proximity model

Two entry points:

```python
prox = model.prox("fingerprint")                    # from a trained model
prox = fs.prox("fingerprint", target="logS")        # from a FeatureSet (pre-model)
prox = fs.prox("features", feature_list=["mollogp", "tpsa"], target="logS")
```

- **`model.prox(space)`** — precomputed-first: returns the proximity the model already
  carries (a fingerprint proximity **frozen at training time** over the training rows —
  what it uses for its own uncertainty). If it has none, builds one **fresh** from the
  model's FeatureSet using the model's own features/target, and logs that it did. Cached
  per `space`.
  - Returns **`None`** for `"features"` when the model's features are structural (contain
    `smiles` or `fingerprint`) — that's a structure model, so use `model.prox("fingerprint")`.
- **`fs.prox(space, …)`** — always builds fresh over the full FeatureSet at the current
  config. The pre-model path: hunt anomalies and neighbors before training. `feature_list`
  is required for `"features"`. Cached per `(space, feature_list, target)`.

`space` is `"fingerprint"` (Tanimoto over SMILES) or `"features"` (Euclidean over
descriptors); `prox.space` reports which you got.

Passing a **target** unlocks target-aware analysis on top of the proximity —
`ActivityLandscape(prox)` (activity cliffs, isolated compounds) and `ResidualFeatures(prox)`
— and adds the neighbor's target to results. Without a target you get plain nearest-neighbors.

For a one-off DataFrame with no artifact, construct a backend directly:

```python
FingerprintProximity(df, id_column="id")                    # a 'smiles' or 'fingerprint' column
FeatureSpaceProximity(df, id_column="id", features=[...])
```

## Neighbors of a compound in the set

```python
nbrs = prox.neighbors(compound_id, n_neighbors=6, include_self=False)
```

One row per neighbor:

| column | meaning |
|---|---|
| `prox.id_column` | the query id (name varies by model — e.g. `id`, `molecule_name`) |
| `neighbor_id` | the neighboring compound |
| `similarity` / `distance` | Tanimoto similarity 0–1 (FingerprintProximity) or standardized Euclidean distance (FeatureSpaceProximity) |
| `<target>` | the neighbor's target value (may be `NaN` for a multi-task model's off-task rows) |

Two gotchas:

- **`include_self` defaults to `True`.** A compound is its own closest neighbor, so
  pass `include_self=False` to get real neighbors.
- **`n_neighbors` counts before self is dropped.** With `include_self=False`, request
  `k + 1` to get `k` neighbors back.

Don't hardcode the query id column — it's `prox.id_column`. `neighbor_id` is stable.

## Neighbors of a novel compound

For a compound **not** in the set, look it up by structure/features instead:

```python
nbrs = prox.neighbors_from_query_df(query_df, n_neighbors=5)
```

`query_df` needs a `smiles` (or `fingerprint`) column for FingerprintProximity, or the
model's feature columns for FeatureSpaceProximity. Result columns are
`query_id, neighbor_id, similarity|distance, <target>`. Unparseable SMILES are dropped
with a warning rather than raising.

## Controlling the search

```python
prox.neighbors(cid, min_similarity=0.5, include_self=False)   # FingerprintProximity: all >= 0.5 Tanimoto
prox.neighbors(cid, radius=2.0, include_self=False)           # FeatureSpaceProximity: all within distance
```

The threshold overrides `n_neighbors` — use it when "how many are actually close"
matters more than a fixed count.

## Putting it together

Neighbors + the structure grid (`plotting` guide) is the activity-cliff view: pull a
high-residual compound's neighbors, then render the query and its neighbors side by
side, captioned with activity. Near-identical scaffolds at opposite ends of the potency
scale is the cliff. SMILES/id column conventions are in the `compounds` guide.
