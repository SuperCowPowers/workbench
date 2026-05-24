"""Find Workbench models whose bundled inference script still imports workbench_bridges.

Background:
    workbench-bridges was merged into the main `workbench` package. New model
    bundles import from `workbench.endpoints.*` instead. Existing models in the
    registry, however, still carry their original `generated_model_script.py`
    with `import workbench_bridges` references — those will keep working as
    long as the inference image continues to ship `workbench-bridges==0.2.10`
    as a backward-compat shim.

    This script sweeps all models via the Workbench API, opens each bundle's
    `generated_model_script.py` (streamed from S3, no local extraction), and
    reports which ones still reference workbench_bridges. Once the report is
    empty (or the listed models are deemed obsolete), the bridges shim can be
    dropped from the next image revision.

Usage:
    AWS_PROFILE=<profile> python scripts/admin/find_models_with_bridges.py
    AWS_PROFILE=<profile> python scripts/admin/find_models_with_bridges.py --model aqsol-reg-pytorch
"""

import argparse
import io
import re
import tarfile
from urllib.parse import urlparse

import boto3

from workbench.api import Meta, Model


# Match `import workbench_bridges` or `from workbench_bridges...`
_BRIDGES_PATTERN = re.compile(rb"^\s*(?:import\s+workbench_bridges|from\s+workbench_bridges)", re.MULTILINE)

# Files inside the model.tar.gz worth scanning. generated_model_script.py is
# the canonical entry point for Workbench-trained models.
_BUNDLE_SCRIPT_NAMES = ("generated_model_script.py",)


def scan_bundle_for_bridges(s3_client, model_data_url: str) -> list[str]:
    """Return bundle filenames that import workbench_bridges.

    Streams the model.tar.gz from S3 into memory (bundles are typically a few
    hundred MB at worst — manageable for an admin sweep). Inspects only files
    whose basenames are in ``_BUNDLE_SCRIPT_NAMES``.

    Args:
        s3_client: A boto3 S3 client.
        model_data_url: S3 URL of the model.tar.gz bundle.

    Returns:
        List of matching filenames (empty if no workbench_bridges references found).
    """
    parsed = urlparse(model_data_url)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    raw = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()

    matches = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            basename = member.name.rsplit("/", 1)[-1]
            if basename not in _BUNDLE_SCRIPT_NAMES:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            if _BRIDGES_PATTERN.search(fh.read()):
                matches.append(member.name)
    return matches


def find_models_with_bridges(model_filter: str | None = None) -> list[dict]:
    """Sweep all Workbench models and report any whose bundle imports workbench_bridges.

    Args:
        model_filter: Optional model name to restrict the sweep to one model group.

    Returns:
        List of dicts with keys: model, model_data_url, matches.
    """
    # Use a plain boto3 session for S3 reads — no role assumption needed for
    # this read-only sweep, and Model.model_data_url() already gives us absolute
    # S3 paths so we don't need anything fancier.
    s3_client = boto3.client("s3")

    meta = Meta()
    models_df = meta.models()
    model_names = models_df["Model Group"].tolist()
    if model_filter:
        model_names = [m for m in model_names if m == model_filter]
        if not model_names:
            print(f"No model named '{model_filter}' found.")
            return []

    findings = []
    for model_name in model_names:
        model = Model(model_name)
        model_data_url = model.model_data_url()
        if not model_data_url:
            print(f"  [skip] {model_name}: no model_data_url (training-only artifact?)")
            continue

        try:
            matches = scan_bundle_for_bridges(s3_client, model_data_url)
        except Exception as e:
            print(f"  [warn] {model_name}: failed to scan bundle ({e})")
            continue

        if matches:
            findings.append({"model": model_name, "model_data_url": model_data_url, "matches": matches})
            print(f"  [hit]  {model_name}: {', '.join(matches)}")
        else:
            print(f"  [ok]   {model_name}")

    print(f"\nScanned {len(model_names)} model(s); {len(findings)} still import workbench_bridges.")
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find Workbench models whose bundles still import workbench_bridges."
    )
    parser.add_argument(
        "--model",
        help="Restrict the sweep to a single model name (default: scan every model).",
    )
    args = parser.parse_args()

    findings = find_models_with_bridges(model_filter=args.model)

    if findings:
        print("\nModels needing migration (or retraining) before workbench-bridges can be dropped:\n")
        for f in findings:
            print(f"  {f['model']}")
            print(f"    data:  {f['model_data_url']}")
            print(f"    files: {', '.join(f['matches'])}")
            print()
    else:
        print("\nAll clear — no models reference workbench_bridges.")
