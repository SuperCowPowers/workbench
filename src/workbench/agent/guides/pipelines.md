# ML Pipelines

> declared artifact DAGs, their on-disk definitions, launching them

A pipeline is a **declared DAG of artifacts** — which scripts produce which
DataSources, FeatureSets, Models, and Endpoints. It is a definition on disk, not
a running service: the DAG describes the dependency graph, and a launcher runs
the scripts that build it.

## Where they live

Under `ML_PIPELINES_ROOT` (see `config`). The directory tree *is* the grouping:

```
ml_pipelines/
  OpenADMET/
    hlm_clint/
      pipelines.json      # the definition
      hlm_clint.py        # the script it runs
  Testing/
    AqSol/ ...
```

## The definition

`pipelines.json` maps a pipeline name to a list of steps:

```json
{
  "pipelines": {
    "hlm_clint": [
      {
        "script": "hlm_clint.py",
        "inputs":  ["fs:open_admet_hlm_clint"],
        "outputs": ["model:hlm-clint-reg-xgb", "endpoint:hlm-clint-reg-xgb"]
      },
      {
        "script": "workbench:models/model_promotion.py",
        "inputs":  ["model:hlm-clint-reg-xgb"],
        "outputs": ["endpoint:hlm-clint-reg-v1"]
      }
    ]
  }
}
```

- Artifact ids are `type:name`, where type is `ds`, `fs`, `model`, `endpoint`,
  or `public` (a public dataset, e.g. `public:comp_chem/logd/logd_all`).
- A step's `inputs`/`outputs` are what create the edges — the DAG is derived, not
  written by hand.
- `script` is a path next to the JSON, or `workbench:...` for a built-in plugin
  script (e.g. model promotion).

Dependencies must follow the artifact chain: `ds -> fs -> model -> endpoint`. A
FeatureSet never feeds an Endpoint directly; an endpoint always comes from a
model.

## Inspecting from the REPL

```python
from workbench.cached.cached_meta import CachedMeta
groups = CachedMeta().pipelines()
```

The result is **nested groups**, not a flat list — each entry is
`{name, subgroups, pipelines}`, and `subgroups` recurses. Walk it:

```python
def walk(groups, depth=0):
    for g in groups:
        print("  " * depth + f"{g['name']}: {list(g['pipelines'])}")
        walk(g["subgroups"], depth + 1)

walk(groups)
```

Each pipeline is `{"nodes": [...], "links": [...]}` — note the key is **links**,
not "edges":

```python
pl = groups[0]["subgroups"][0]["pipelines"]["hlm_clint"]
{n["type"] for n in pl["nodes"]}      # {'ds', 'fs', 'model', 'endpoint'}
pl["links"][0]                        # {'source': 'ds:...', 'target': 'fs:...'}
```

`networkx` is available if the user wants real graph work (topological order,
ancestors of a node, orphan detection).

## Is this artifact in a pipeline?

**Use the utility — do not hand-roll the search.** The hierarchy nests
(groups contain subgroups) and promoted models carry a date suffix the pipeline
does not declare. Both are easy to miss and produce confident wrong answers.

```python
from workbench.utils.pipeline_utils import find_pipelines

find_pipelines("pxr-reg-chemprop-mt-logd-260715")
# [{'group': 'pxr', 'pipeline': 'openadmet_pxr', 'matched': 'model:pxr-reg-chemprop-mt-logd'}]

find_pipelines("pxr-reg-v1", "endpoint")     # other types: ds, fs, endpoint, public
```

The `matched` field shows which node id actually hit, so you can tell the user
whether it matched directly or via the de-promoted base name.

### Promoted models carry a date suffix

Promotion copies the winning model to `<base-name>-YYMMDD` and hands that copy
the endpoint. The dated copy is the champion; the **base name** is what the
pipeline declares.

```
model:pxr-reg-chemprop-mt-logd            <- declared pipeline node
pxr-reg-chemprop-mt-logd-260715           <- promoted champion, 2026-07-15
```

`find_pipelines()` strips the suffix for you. A promoted model **is** part of
its pipeline — it is the winner that pipeline produced. Saying "not in a
pipeline" because the dated name isn't a node is wrong.

An empty result is still a real answer: models built ad hoc in a REPL session
are in no pipeline, and that is normal.

## Show me the pipelines

For "show me the pipelines" or a visual look at a DAG, open the dashboard page:

```python
from workbench.utils.dashboard_utils import open_page
open_page("ml_pipelines")
```

## Running one

Pipelines are launched from the shell, not the REPL:

```bash
ml_pipeline_launcher hlm_clint --dry-run   # always look first
ml_pipeline_launcher hlm_clint
```

Useful flags: `--dry-run`, `--all`, `--full-dag`, `--local`, `--realtime`
(serverless is the default).

Launching rebuilds real artifacts and can retrain many models, so treat it as
the user's call: show them the `--dry-run` output and hand them the command
rather than running it yourself.
