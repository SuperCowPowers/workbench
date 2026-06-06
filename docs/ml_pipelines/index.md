# ML Pipelines

An **ML pipeline** is a set of scripts that build Workbench artifacts (FeatureSets,
Models, Endpoints) with dependencies between them. Pipelines are declared in a
`pipelines.json` file and launched with the `ml_pipeline_launcher` CLI, which
orders the runs and submits them to AWS Batch (or runs them locally).

## pipelines.json

The launcher discovers every `pipelines.json` under the current directory. Each
file maps pipeline names to a **flat list of nodes**:

```json
{
  "pipelines": {
    "wine_classifier": [
      {"script": "wine_classifier.py"}
    ],
    "aqsol": [
      {"script": "aqsol_feature_set.py", "outputs": ["fs:aqsol_features"]},
      {"script": "aqsol_class.py", "inputs": ["fs:aqsol_features"]},
      {"script": "aqsol_reg.py", "inputs": ["fs:aqsol_features"]}
    ]
  }
}
```

### Node schema

| Field     | Required | Description                                                        |
|-----------|----------|--------------------------------------------------------------------|
| `script`  | yes      | Script filename, resolved relative to the `pipelines.json` it's in. |
| `mode`    | no       | Run mode, `dt` or `ts`. Omit for a modeless node (see [Modes](#modes)). |
| `outputs` | no       | Artifact refs this node produces, e.g. `["fs:aqsol_features"]`.     |
| `inputs`  | no       | Artifact refs this node depends on.                                |

### Artifact refs

Dependencies are expressed through **artifacts**, not by naming other scripts. An
artifact ref is a `type:name` string:

| Prefix   | Artifact   |
|----------|------------|
| `fs:`    | FeatureSet |
| `ds:`    | DataSource |
| `model:` | Model      |

The execution graph is **derived**: for each `inputs` ref, the launcher draws an
edge from whichever node lists it in `outputs`. You never hand-write the ordering —
it falls out of the data flow. The launcher prints each pipeline as a tree —
producers at the root, the consumers beneath them. The `aqsol` pipeline above
resolves to:

```
aqsol
╙── aqsol_feature_set
    ├─╼ aqsol_class
    └─╼ aqsol_reg
```

In a terminal this is colorized to make the structure easy to scan: pipeline names,
the `[dt]`/`[ts]` mode tags, and leaf (terminal) nodes are tinted. Colors are
suppressed automatically when output isn't a TTY (piped or redirected).

### Modes

A node may declare a singular `mode`. Because the artifacts a script produces can
differ by mode, a script that runs in more than one mode appears as **more than one
node**:

```json
{
  "pipelines": {
    "caco2_er_reg": [
      {"script": "caco2_er_reg_xgb_1.py", "mode": "dt", "outputs": ["fs:caco2_er_f1"]},
      {"script": "caco2_er_reg_xgb_1.py", "mode": "ts", "inputs": ["fs:caco2_er_f1"]}
    ]
  }
}
```

Here the `ts` run depends on the `dt` run's FeatureSet. Mode selection at launch time:

- **`--dt` / `--ts`** — run only nodes with that mode. Modeless nodes always run.
- **no mode flag** — run every node (all modes), in dependency order.
- **`--promote` / `--test-promote`** — run each unique script once, ignoring modes
  and edges.

When you launch `--ts` alone, a `ts` node's input artifact has no producer in that
run; the launcher assumes the `dt` run already created it and the job proceeds
(depending on the `dt` job automatically if it happens to still be running).

## Validation

When a selected pipeline is built, the launcher enforces:

- **One producer per artifact** — two nodes listing the same ref in `outputs` is an error.
- **Acyclic** — a dependency cycle is an error.
- **Dangling input** — an `inputs` ref that no node outputs emits a warning and is
  treated as an external (already-existing) input.

## Running pipelines

Run `ml_pipeline_launcher` from a directory containing your pipeline scripts and
`pipelines.json`:

```bash
ml_pipeline_launcher --dt --all            # Launch ALL pipelines in DT mode
ml_pipeline_launcher --dt caco2            # Launch pipelines matching 'caco2' in DT mode
ml_pipeline_launcher --ts --all            # Temporal-split ALL pipelines
ml_pipeline_launcher --promote --all       # Promote ALL pipelines
ml_pipeline_launcher caco2                 # Launch matching pipelines in their declared modes
ml_pipeline_launcher --dt --dry-run        # Show what would launch, without launching
ml_pipeline_launcher --local --dt aqsol    # Run a single pipeline locally
```

Args after a literal `--` are forwarded verbatim to each underlying script:

```bash
ml_pipeline_launcher --dt my_script.py -- --epochs 10 --lr 0.01
```

A full runnable example lives in
[`examples/ml_pipelines/`](https://github.com/SuperCowPowers/workbench/tree/main/examples/ml_pipelines).
