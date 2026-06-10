# ci/

Scripts that back the repository's GitHub Actions checks. Each one is runnable
locally with the same invocation CI uses, so you can reproduce a failure without
pushing.

| Script | What it does | Enforced by |
| --- | --- | --- |
| [`lock.sh`](lock.sh) | Regenerates every image's `requirements.lock` from its inputs | [`lockfile-drift.yml`](../.github/workflows/lockfile-drift.yml) |
| [`endpoint_import_smoke.py`](endpoint_import_smoke.py) | Checks every `workbench.endpoints` module imports under the lean endpoint dep surface | [`endpoint-import-smoke.yml`](../.github/workflows/endpoint-import-smoke.yml) |
| [`endpoint_smoke_requirements.txt`](endpoint_smoke_requirements.txt) | The lean dep manifest the smoke test installs against (intersection of every deployed endpoint container's requirements) | — |

## `lock.sh` — lockfile regeneration

The workspace `uv.lock` is the **master**; the per-image `requirements.lock`
files are derived views. `lock.sh` exports `uv.lock` as a constraints file and
compiles each image's lock from its inputs (`pyproject.toml` plus, where
present, a per-image `requirements.in` for image-only deps like
`fastapi`/`uvicorn`) under those constraints — so wherever an image shares a
package with the workspace, the versions match.

### Moving the master and pushing to the images

```bash
uv lock --upgrade        # or: uv lock --upgrade-package <name>
./ci/lock.sh             # propagate to the image locks
git diff                 # review, then commit uv.lock + *.lock together
```

The same two steps apply after editing a dependency in `pyproject.toml` or an
image's `requirements.in` (skip `--upgrade` — a plain `uv lock` picks up the
edit).

### Resolution behavior

New upstream releases on PyPI do **not** move the locks — versions move only
when `uv.lock` moves or a dependency constraint changes. Packages outside the
workspace are unconstrained by the master: image-only overlays like
`chemprop`/`shap`, plus torch/`nvidia-*`/`triton`, which are deliberately
excluded from the constraints because the images pin index-specific variants
(`+cpu`/`+cu130`). Those re-pin incrementally, reusing each lock's existing
versions as preferred-version hints; `UPGRADE=1 ./ci/lock.sh` drops the hints
and refreshes them to the latest.

### CI gate

[`lockfile-drift.yml`](../.github/workflows/lockfile-drift.yml) runs `./ci/lock.sh`
and fails if `git diff` shows any change, catching a `uv.lock` /
`pyproject.toml` / `requirements.in` change that wasn't propagated to the image
locks. The check stays green as upstream releases happen — a diff only appears
when the master or a dependency constraint actually moved. To fix a failure:
run `./ci/lock.sh` locally and commit the diff.

## `endpoint_import_smoke.py` — endpoint import contract

Enumerates every module under `workbench/endpoints/` and verifies each imports
cleanly against the lean endpoint dep surface
([`endpoint_smoke_requirements.txt`](endpoint_smoke_requirements.txt)) without
pulling in any heavy orchestration/UI/analysis library transitively. A failure
here means an endpoint container would also fail on its first cold-start after
the change deploys.

```bash
pip install -r ci/endpoint_smoke_requirements.txt
pip install --no-deps -e .
python ci/endpoint_import_smoke.py
```

The script is the single source of truth for coverage: adding a new module under
`workbench/endpoints/` automatically extends the check — no workflow edit needed.
