# PyPI Release Notes

Releases normally happen automatically: pushing a `v*` tag triggers the
`.github/workflows/publish.yml` GitHub Action, which lints, builds,
publishes to PyPI via trusted publishing (OIDC), and creates a GitHub Release. These
notes cover doing the same thing **manually from a laptop** when the Action is
unavailable or you need to bypass it.

For full details on packaging see the
[Packaging tutorial](https://packaging.python.org/tutorials/packaging-projects/#packaging-your-project).

> **Heads up — the tag triggers the Action.** `git push --tags` is what fires
> `publish.yml`. If the tag is already pushed, the Action is already publishing that
> version, so don't also publish manually — you'll just get a "file already exists" 400
> from PyPI. Do a manual release either (a) as a **fallback** after the Action failed on
> an already-pushed tag, or (b) as a **full bypass** where you build and upload *before*
> pushing the tag.

### Package Requirements

```bash
pip install --upgrade build twine
```

### How the version is set

The version is derived from the git tag by `setuptools_scm` (see `[tool.setuptools_scm]`
in `pyproject.toml`) — there is no hardcoded version string. The build must run from a
**tagged commit**, otherwise you get a `.devN` version instead of the clean release
version.

### Set up ~/.pypirc

The Action uses OIDC trusted publishing and needs no token. A manual upload from your
laptop still needs an API token. Put it in `~/.pypirc`:

```ini
[distutils]
index-servers =
  pypi
  testpypi

[pypi]
username = __token__
password = pypi-AgEIcH...

[testpypi]
username = __token__
password = pypi-AgENdG...
```

### Lint (matches the Action's gate)

The Action publishes only if lint passes, so run the same checks locally first:

```bash
black --check --line-length=120 src/workbench applications tests
flake8 --exclude '*generated*' src/workbench applications tests
```

Optionally run the full test suite via tox, which installs the built package into a clean
virtualenv and runs the tests against it:

```bash
tox
```

### Clean previous distribution files

```bash
make clean
```

### Tag the new version

```bash
git tag v0.1.8   # or whatever the next version is
```

Push the tag **now** only if you want the GitHub Action to do the release. For a manual
bypass, hold off on pushing until after the upload (see the last step).

### Build

`setuptools_scm` reads the version from the tag you just created.

```bash
python -m build
```

### (Optional) Test PyPI dry run

```bash
twine upload dist/* -r testpypi
pip install --index-url https://test.pypi.org/simple workbench
```

### Publish to PyPI

```bash
twine upload dist/* -r pypi
```

### Push the tag and any changes

```bash
git push
git push --tags
```

If you published manually as a **full bypass**, pushing the tag here will still trigger
the Action. It will fail at the PyPI upload step (version already exists) but will still
create the GitHub Release — or create that release yourself:

```bash
gh release create v0.1.8 dist/* --title v0.1.8 --generate-notes
```
