# CI

Workbench CI has two parts: dependency locking (one master, propagated to the
image lockfiles) and the GitHub Actions that gate every push. Everything is
runnable locally — see [`ci/README.md`](https://github.com/SuperCowPowers/workbench/blob/main/ci/README.md)
for the detailed runbook.

## Dependency Locking: uv lock → propagate

The workspace [`uv.lock`](https://github.com/SuperCowPowers/workbench/blob/main/uv.lock)
is the **master** for all pinned versions. The seven per-image
`requirements.lock` files (dashboard, compound explorer, ml_pipelines, base
training/inference, pytorch_chem training/inference) are **derived views**:
[`ci/lock.sh`](https://github.com/SuperCowPowers/workbench/blob/main/ci/lock.sh)
exports `uv.lock` as a constraints file and compiles each image's lock from
`pyproject.toml` plus its `requirements.in` overlay under those constraints, so
shared packages match the workspace everywhere.

To move the master and push to the images:

```bash
uv lock --upgrade        # or: uv lock --upgrade-package <name>
./ci/lock.sh             # propagate to the image locks
git diff                 # review, then commit uv.lock + *.lock together
```

Only packages outside the workspace escape the master: image-only overlays
(`chemprop`, `shap`, ...) and the torch/`nvidia-*`/`triton` index variants
(`+cpu`/`+cu130`), which each image pins deliberately.

## GitHub Actions

| Workflow | Trigger | What it enforces |
| --- | --- | --- |
| [`python-lint.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/python-lint.yml) | push, PR | Black (line length 120) and Flake8 over `src/workbench`, `applications`, `tests` |
| [`lockfile-drift.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/lockfile-drift.yml) | push, PR | Reruns `ci/lock.sh`; fails if any `requirements.lock` is out of sync with `uv.lock` / `pyproject.toml` / `requirements.in` |
| [`endpoint-import-smoke.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/endpoint-import-smoke.yml) | push, PR | Every `workbench.endpoints` module imports under the lean endpoint dep surface (no heavy orchestration/UI libs pulled in transitively) |
| [`deploy-docs.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/deploy-docs.yml) | push to `main` | Builds this MkDocs site and deploys it to GitHub Pages |
| [`publish.yml`](https://github.com/SuperCowPowers/workbench/blob/main/.github/workflows/publish.yml) | `v*` tag | Lints, builds, publishes to PyPI (trusted publishing), and creates a GitHub release with changelog notes |

## Fixing a Red Check

- **Lockfile Drift Check** — run `./ci/lock.sh` and commit the diff. If you
  changed `uv.lock` (or `pyproject.toml`), the lock diff belongs in the same
  commit.
- **Endpoint Import Surface Smoke Test** — an endpoint module now imports
  something outside the lean dep manifest
  ([`ci/endpoint_smoke_requirements.txt`](https://github.com/SuperCowPowers/workbench/blob/main/ci/endpoint_smoke_requirements.txt)).
  Either make the import lazy or reconsider the dependency; a failure here
  means the endpoint container would fail on cold-start.
- **Python Linting** — `black --line-length=120 src/workbench applications tests`
  then fix any remaining Flake8 complaints.
