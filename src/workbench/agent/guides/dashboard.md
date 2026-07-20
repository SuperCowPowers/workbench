# The Workbench Dashboard

> the web dashboard, its pages, plugins, and artifact deep links

A Dash/Plotly web app over the **same artifacts** the REPL works with — no
separate data, no sync step. A model you create in the REPL shows up in the
dashboard; the dashboard is the visual view, the REPL is the programmatic one.

You cannot drive the dashboard from here. When a user wants to browse
visually, compare models side by side, or share something with a colleague,
point them at it rather than trying to reproduce it in text.

## Linking to an artifact

Each artifact page takes a `?name=` query parameter, so you can hand the user a
clickable deep link instead of describing where to click:

```python
from workbench.utils.dashboard_utils import artifact_url, open_artifact

artifact_url("model", "aqsol-fingerprint-reg-v1")   # just the link
# https://workbench-dashboard.com/models?name=aqsol-fingerprint-reg-v1

open_artifact("model", "aqsol-fingerprint-reg-v1")  # opens a browser, returns the URL
```

Types: `data_source`, `feature_set`, `model`, `endpoint`, `pipeline`.

Which to use: **"show me / open / pull up X in the dashboard" → `open_artifact()`**,
then say what you opened. If they just want a link to keep or share, or you're
mentioning the page in passing, use `artifact_url()` and print it. Don't launch
a browser the user didn't ask for.

The base URL comes from the `DASHBOARD_URL` config value, which **returns None
when unset** — it is not discoverable from AWS. If you get None, don't guess or
fabricate a URL; either say nothing about links or mention that `DASHBOARD_URL`
can be set in the Workbench config.

When it is configured, offer a link whenever you name a specific artifact the
user might want to look at.

## Pages

- **main** — overview of everything in the account
- **data_sources / feature_sets / models / endpoints** — one page per artifact
  type, with details, metrics, and plots
- **ml_pipelines** — the pipeline DAGs
- **contests** — model comparison reports
- **status / license** — health and licensing

## Running it locally

```bash
cd applications/aws_dashboard
pip install -r requirements.txt
python app.py          # or ./dashboard
```

It reads the same Workbench config as the REPL, so it points at whatever AWS
account the current config selects.

Deployed instances come from the Dashboard CDK stack — that is an admin/infra
task, not something to do from a REPL session.

## Plugins

The dashboard is meant to be extended. Point `WORKBENCH_PLUGINS` at a directory
(local path or `s3://...`) and everything under it loads at startup:

```
plugins/
  components/   # subclass PluginInterface -- auto-loads onto an artifact page
  pages/        # a class with page_setup(app) -- registers its own route
  views/        # subclass PageView -- reshape the data behind a page
  assets/       # clientside JS/CSS, served and injected by Dash
  packages/     # importable Python packages for your plugins
```

A component sets `auto_load_page` (which page it attaches to) and
`plugin_input_type` (what object it receives), then implements create/update.
This is how clients add business-specific views without forking the dashboard.

## Relationship to the REPL

The REPL's `plot()` command renders using dashboard components, so a plot you
get in the REPL and the dashboard's version are the same code. For quick
matplotlib work inside a session, see the `plotting` guide instead.

## More

- Dashboard setup: https://supercowpowers.github.io/workbench/aws_setup/dashboard_stack/
- Plugins: https://supercowpowers.github.io/workbench/plugins/
- S3 plugins: https://supercowpowers.github.io/workbench/admin/dashboard_s3_plugins/
