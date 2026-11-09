"""Find models whose bundled inference script no longer matches the workbench in the images.

A model bundle carries the ``generated_model_script.py`` that was filled from the
template at training time, while the serving container supplies ``workbench`` from
the image's site-packages. The image tag floats (``:latest``), so a bundle trained
against an older workbench can lose a symbol it imports and the endpoint dies at
startup with ``ImportError`` -> "the model process exited".

This sweep opens each bundle's script, resolves every ``workbench.*`` import against
the installed workbench, and reports the bundles whose imports no longer resolve.
Those models need a retrain to re-bake the script; the script also prints the
``ml_pipeline_launcher`` commands that do it, resolved from the ``pipelines.json``
files that declare each model as an output.

Scope is the DT models and the promoted copies taken from them -- only DT models get
promoted, so a stale TS bundle never reaches a served endpoint. ``--model`` overrides
the filter for a one-off check.

The verdict is only as good as the installed workbench matching the images, so the
header prints both and warns on a mismatch.

Usage:
    AWS_PROFILE=<profile> python scripts/admin/find_stale_model_bundles.py
    AWS_PROFILE=<profile> python scripts/admin/find_stale_model_bundles.py --model cl-mouse-reg-chemprop-1-dt
    AWS_PROFILE=<profile> python scripts/admin/find_stale_model_bundles.py --pipelines-dir ~/work/ideaya/ml_pipelines
"""

import argparse
import ast
import importlib
import importlib.metadata
import json
import re
import tarfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3

from workbench.api import Meta, Model

# The bundle's inference entry point -- the only member we need out of the tarball.
BUNDLE_SCRIPT = "generated_model_script.py"

# The suffix a DT run stamps onto a model name, and the date stamp a promoted copy
# carries ("cl-mouse-reg-chemprop-1-261109").
DT_SUFFIX = "-dt"
DATE_STAMP_RE = re.compile(r"-\d{6}$")

# The image Dockerfile pinning the workbench the endpoints actually serve.
INFERENCE_DOCKERFILE = Path(__file__).resolve().parents[2] / "sagemaker_images/pytorch_chem/inference/Dockerfile"
WORKBENCH_VERSION_RE = re.compile(r"^ARG WORKBENCH_VERSION=(\S+)", re.MULTILINE)


def image_workbench_version() -> str | None:
    """Workbench version pinned in the inference image, or None when the repo isn't at hand."""
    try:
        return WORKBENCH_VERSION_RE.search(INFERENCE_DOCKERFILE.read_text()).group(1)
    except (OSError, AttributeError):
        return None


def read_bundle_script(s3_client, model_data_url: str) -> str | None:
    """Return the bundle's generated_model_script.py source, or None when it has none.

    Streams the tarball (``r|gz``) and stops at the script, so a 40MB bundle costs
    only the leading members rather than a full download.
    """
    parsed = urlparse(model_data_url)
    body = s3_client.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
    with tarfile.open(fileobj=body, mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.rsplit("/", 1)[-1] == BUNDLE_SCRIPT:
                fh = tar.extractfile(member)
                return fh.read().decode("utf-8", "replace") if fh else None
    return None


def unresolved_workbench_imports(source: str) -> list[str]:
    """Return 'module.name' for every workbench import in the script that no longer resolves.

    Covers both a module that's gone and a name the module no longer exports. Imports
    inside functions count too -- the template defers training-only imports, but a
    missing one still breaks that code path.
    """
    missing = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module, names = node.module, [a.name for a in node.names]
        elif isinstance(node, ast.Import):
            module, names = None, [a.name for a in node.names]
        else:
            continue
        for mod, name in [(module, n) for n in names] if module else [(n, None) for n in names]:
            if not mod or not (mod == "workbench" or mod.startswith("workbench.")):
                continue
            try:
                imported = importlib.import_module(mod)
            except ImportError:
                missing.append(f"{mod}.{name}" if name else mod)
                continue
            if name and name != "*" and not hasattr(imported, name):
                # A submodule not yet imported isn't an attribute of its package yet.
                try:
                    importlib.import_module(f"{mod}.{name}")
                except ImportError:
                    missing.append(f"{mod}.{name}")
    return missing


def model_producers(pipelines_dir: Path) -> dict[str, set[tuple[str, str | None]]]:
    """Map each declared model name to the (script stem, mode) jobs that output it.

    Reads every pipelines.json under ``pipelines_dir``. The stem is what
    ml_pipeline_launcher matches its patterns against, and the job's own "mode" is
    the flag that selects it.
    """
    producers = defaultdict(set)
    for spec_path in sorted(pipelines_dir.rglob("pipelines.json")):
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [warn] {spec_path}: unreadable ({e})")
            continue
        for jobs in spec.get("pipelines", {}).values():
            for job in jobs:
                stem = Path(str(job.get("script", ""))).name.removesuffix(".py")
                if not stem:
                    continue
                for ref in job.get("outputs", []):
                    kind, _, name = ref.partition(":")
                    if kind == "model":
                        producers[name].add((stem, job.get("mode")))
    return producers


def resolve_run(model_name: str, producers: dict[str, set[tuple[str, str | None]]]) -> set[tuple[str, str | None]]:
    """Return the (script stem, mode) jobs that rebuild this registry model.

    A spec declares the stamped name a run produces ("model:cl-mouse-reg-chemprop-1-dt"),
    so match it directly. A promoted copy carries a date stamp instead
    ("cl-mouse-reg-chemprop-1-261109"); rebuilding it means retraining the DT challenger
    it was copied from, so fall back to that name.
    """
    if model_name in producers:
        return producers[model_name]
    base = DATE_STAMP_RE.sub("", model_name)
    if base != model_name:
        return producers.get(f"{base}{DT_SUFFIX}", set())
    return set()


def scan_models(model_names: list[str], workers: int) -> list[dict]:
    """Scan each model's bundle and return a finding per model with unresolved imports."""
    s3_client = boto3.client("s3")

    def scan(model_name: str) -> dict | None:
        model_data_url = Model(model_name).model_data_url()
        if not model_data_url:
            print(f"  [skip] {model_name}: no model_data_url")
            return None
        try:
            source = read_bundle_script(s3_client, model_data_url)
        except Exception as e:
            print(f"  [warn] {model_name}: failed to read bundle ({e})")
            return None
        if source is None:
            print(f"  [skip] {model_name}: bundle has no {BUNDLE_SCRIPT}")
            return None
        try:
            missing = unresolved_workbench_imports(source)
        except SyntaxError as e:
            print(f"  [warn] {model_name}: unparseable script ({e})")
            return None
        if missing:
            print(f"  [stale] {model_name}: {', '.join(missing)}")
            return {"model": model_name, "missing": missing, "model_data_url": model_data_url}
        print(f"  [ok]    {model_name}")
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [f for f in pool.map(scan, model_names) if f]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", help="Restrict the sweep to one model (default: every model)")
    parser.add_argument(
        "--pipelines-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory holding the pipelines.json files that declare the models (default: cwd)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent bundle reads (default: 8)")
    args = parser.parse_args()

    installed = importlib.metadata.version("workbench")
    pinned = image_workbench_version()
    print(f"Installed workbench: {installed}   inference image pins: {pinned or 'unknown'}")
    # A local editable install carries a ".devN" suffix off the same release, which
    # tracks the image fine -- only a differing release is worth a warning.
    if pinned and pinned != installed.partition(".dev")[0]:
        print("  [warn] versions differ -- the imports resolve against the installed one, not the image's")

    all_models = Meta().models()["Model Group"].tolist()
    if args.model:
        model_names = [m for m in all_models if m == args.model]
        if not model_names:
            print(f"No model named '{args.model}' found.")
            return
    else:
        model_names = [m for m in all_models if m.endswith(DT_SUFFIX) or DATE_STAMP_RE.search(m)]
        print(f"\n{len(model_names)} of {len(all_models)} models are DT or promoted copies")

    print(f"\nScanning {len(model_names)} model bundle(s)...")
    findings = scan_models(model_names, args.workers)
    print(f"\n{len(findings)} of {len(model_names)} bundle(s) have workbench imports that no longer resolve.")
    if not findings:
        print("All clear -- every bundle's script matches the installed workbench.")
        return

    producers = model_producers(args.pipelines_dir)
    by_mode = defaultdict(set)
    unmatched = []
    for finding in findings:
        jobs = resolve_run(finding["model"], producers)
        if jobs:
            for stem, mode in jobs:
                by_mode[mode].add(stem)
        else:
            unmatched.append(finding["model"])

    if by_mode:
        print(f"\nRetrain commands (pipelines from {args.pipelines_dir}):\n")
        for mode in sorted(by_mode, key=lambda m: (m is None, m)):
            flag = f"--{mode} " if mode else ""
            print(f"  ml_pipeline_launcher {flag}{' '.join(sorted(by_mode[mode]))}")

    if unmatched:
        print("\nNo pipeline declares these as an output -- retrain or retire them by hand:\n")
        for model_name in unmatched:
            print(f"  {model_name}")


if __name__ == "__main__":
    main()
