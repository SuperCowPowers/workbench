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
it falls out of the data flow. Internally this is a **bipartite DAG** (the
Dagster/Airflow model): artifacts *and* scripts are both nodes, with edges flowing
`input-artifact → script → output-artifact`. The launcher prints it as a tree, so
the `aqsol` pipeline (with a `ds:aqsol_data` source and `model:` outputs) resolves to:

```
ds:aqsol_data
└─╼ aqsol_feature_set
    └─╼ fs:aqsol_features
        ├─╼ aqsol_class ─╼ model:aqsol-class
        └─╼ aqsol_reg ─╼ model:aqsol-reg
```

In a terminal this is colorized to make the structure easy to scan: script names,
the `[dt]`/`[ts]` mode tags, and dependency artifacts (tagged with their freshness —
see [Freshness](#freshness)) are tinted. Colors are suppressed automatically when
output isn't a TTY (piped or redirected).

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
- **`--promote`** — run each unique script once, ignoring modes and edges.

When you launch `--ts` alone, a `ts` node's input artifact has no producer in that
run; the launcher assumes the `dt` run already created it and the job proceeds
(depending on the `dt` job automatically if it happens to still be running).

## Validation

When a selected pipeline is built, the launcher enforces:

- **One producer per artifact** — two nodes listing the same ref in `outputs` is an error.
- **Acyclic** — a dependency cycle is an error.
- **Dangling input** — an `inputs` ref that no node outputs is treated, silently, as
  an external (already-existing) input — e.g. a DataSource, or a FeatureSet built by
  another pipeline. It's a graph *root*, not an error.

## Freshness

The launcher is **modification-time aware** (Dagster-style "stale" propagation): it
resolves each artifact's last-modified time from AWS (Glue/SageMaker) and submits a
job only when it's actually out of date. Staleness floods *forward* over the DAG —
if a DataSource changed, its FeatureSet rebuilds, and every model downstream
rebuilds in turn — in one pass, in dependency order. A job is stale when an output
is missing, an input is newer than its outputs, or an upstream job is itself
rerunning.

Two intents are layered on top:

- **Named patterns force a run.** `ml_pipeline_launcher ppb_human` runs the matched
  scripts *regardless* of freshness (you just edited them); their up-to-date
  dependencies stay put. `--all` is freshness-only (rebuild whatever drifted).
- **Dependency freshness is shown.** Each dependency artifact in the printed DAG is
  tagged `(current)` / `(modified)` / `(missing)` so you can see *why* something will
  or won't rerun.

A safety guard: if nearly every artifact comes back missing (the fingerprint of a
wrong AWS account/region), the launcher aborts rather than resubmit the whole world —
override with `--force`.

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
ml_pipeline_launcher --full-dag caco2      # Show the whole closure, incl. up-to-date nodes
ml_pipeline_launcher --sim-mod ds:foo      # Simulate "ds:foo changed" — what would rerun? (offline)
ml_pipeline_launcher --all --force         # Submit even if the wrong-environment guard trips
```

Args after a literal `--` are forwarded verbatim to each underlying script:

```bash
ml_pipeline_launcher --dt my_script.py -- --epochs 10 --lr 0.01
```

A full runnable example lives in
[`examples/ml_pipelines/`](https://github.com/SuperCowPowers/workbench/tree/main/examples/ml_pipelines).

## PipelineManager (the engine)

Both the launcher and the nightly DT Lambda share one class,
[`PipelineManager`](https://github.com/SuperCowPowers/workbench/blob/main/src/workbench/lambda_layer/pipeline_manager.py),
which owns the schema, the bipartite DAG, freshness resolution, and scheduling. Use
it directly to introspect a pipeline tree:

```python
from workbench.lambda_layer.pipeline_manager import PipelineManager

pm = PipelineManager("path/to/ml_pipelines")     # local dir or s3:// prefix
pm.list_pipelines()                                # named pipelines
PipelineManager.show(pm.get_pipeline("aqsol"))     # ascii DAG for one pipeline
PipelineManager.show(pm.upstream_graph("model:aqsol-reg"))    # what produces it
PipelineManager.show(pm.downstream_graph("ds:aqsol_data"))    # what a change impacts

for item in pm.plan():        # (job, run, reason) per job, in topological order
    print(item.job.node_id, item.run, item.reason)
```

It lives under `workbench.lambda_layer` because it ships in the
[workbench Lambda layer](../lambda_layer/index.md) — a dependency-minimal slice of
workbench that the DT Lambda imports to run this *exact* scheduling logic in AWS.
