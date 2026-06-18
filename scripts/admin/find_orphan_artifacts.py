"""Find orphan artifacts -- artifacts that are no longer part of any active pipeline.

Compares every artifact known to Workbench (via CachedMeta) against the set of
artifacts referenced by all pipelines (via PipelineManager). Anything not
referenced is an orphan.

Pipelines only declare ds/fs/model refs (endpoint refs aren't supported), so an
endpoint is considered live when its model (endpoint.get_input()) is a live
pipeline model.

Deletion is intentionally out of scope -- this only reports.
"""

import argparse

from workbench.api import Endpoint
from workbench.cached.cached_meta import CachedMeta
from workbench.lambda_layer.pipeline_manager import PipelineManager, ref_name, ref_type

# Artifacts (any type) whose name contains any of these substrings are never flagged.
EXCLUDE_SUBSTRINGS = ["smiles"]

# DataSource names containing any of these substrings are never flagged.
DS_EXCLUDE_SUBSTRINGS = ["gen_processed"]

# Models/Endpoints whose Owner contains this substring are never flagged
# (e.g. "Pro-BW", "Pro-ER" -- production-owned artifacts).
PROTECTED_OWNER_SUBSTR = "Pro"

# Maps our internal type keys to bulk_delete's item_type strings.
BULK_DELETE_TYPE = {
    "data_sources": "DataSource",
    "feature_sets": "FeatureSet",
    "models": "Model",
    "endpoints": "Endpoint",
}


def _excluded(name: str) -> bool:
    """True if the artifact name matches the global exclude list."""
    return any(sub in name.lower() for sub in EXCLUDE_SUBSTRINGS)


def _ds_excluded(name: str) -> bool:
    """True if a DataSource name matches the global or DataSource-specific exclude list."""
    return _excluded(name) or any(sub in name.lower() for sub in DS_EXCLUDE_SUBSTRINGS)


def _owner_protected(owner: str) -> bool:
    """True if the artifact's Owner marks it as production-owned (never flag)."""
    return bool(owner) and PROTECTED_OWNER_SUBSTR in owner


def find_orphans(pipelines_path: str = ".", full_traverse: bool = False):
    """Find artifacts that aren't referenced by any active pipeline.

    Args:
        pipelines_path: Local dir (or s3:// prefix) holding pipelines.json files.
        full_traverse: If True, resolve each endpoint's model via Endpoint.get_input()
            (accurate but slow). If False (default), assume the endpoint name matches
            its model name.
    """
    meta = CachedMeta()

    # Live set: every artifact ref across all pipelines, split by type.
    pm = PipelineManager(pipelines_path)
    live_refs = {n for n, d in pm.graph.nodes(data=True) if d.get("kind") == "artifact"}
    live = {
        "ds": {ref_name(r) for r in live_refs if ref_type(r) == "ds"},
        "fs": {ref_name(r) for r in live_refs if ref_type(r) == "fs"},
        "model": {ref_name(r) for r in live_refs if ref_type(r) == "model"},
    }
    print(
        f"Loaded {pm.get_num_pipelines()} pipeline(s): "
        f"{len(live['ds'])} ds, {len(live['fs'])} fs, {len(live['model'])} model refs.\n"
    )

    # All artifacts known to Workbench, by type. Models/Endpoints pulled with
    # details=True so we get the Owner field (used to skip production-owned ones).
    all_fs = set(meta.feature_sets()["Feature Group"])
    models_df = meta.models(details=True)
    endpoints_df = meta.endpoints(details=True)
    model_owner = dict(zip(models_df["Model Group"], models_df["Owner"]))
    endpoint_owner = dict(zip(endpoints_df["Name"], endpoints_df["Owner"]))

    orphans = {
        # DataSource orphans are hard to properly identify so we're not going to list them.
        # all_ds = set(meta.data_sources()["Name"])
        # "data_sources": sorted(n for n in all_ds - live["ds"] if not _ds_excluded(n)),
        "feature_sets": sorted(n for n in all_fs - live["fs"] if not _excluded(n)),
        "models": sorted(
            n for n in set(model_owner) - live["model"] if not _excluded(n) and not _owner_protected(model_owner.get(n))
        ),
    }

    # Endpoints: live when their model is a live pipeline model. With full_traverse
    # we resolve the real model via get_input(); otherwise we assume endpoint name
    # == model name (fast, no Endpoint object construction).
    orphan_endpoints = []
    for endpoint_name in sorted(endpoint_owner):
        if _excluded(endpoint_name) or _owner_protected(endpoint_owner.get(endpoint_name)):
            continue
        model = Endpoint(endpoint_name).get_input() if full_traverse else endpoint_name
        if model not in live["model"]:
            orphan_endpoints.append((endpoint_name, model))

    # Report
    for artifact_type, names in orphans.items():
        if names:
            print(f"Orphan {artifact_type} ({len(names)}):")
            for name in names:
                print(f"  {name}")
            print()
        else:
            print(f"No orphan {artifact_type}.\n")

    if orphan_endpoints:
        print(f"Orphan endpoints ({len(orphan_endpoints)}):")
        for endpoint_name, model in orphan_endpoints:
            print(f"  {endpoint_name}  (model: {model})")
        print()
    else:
        print("No orphan endpoints.\n")

    result = {**orphans, "endpoints": [e for e, _ in orphan_endpoints]}

    # bulk_delete-ready block: copy/paste into bulk_delete(orphans_to_delete).
    # Ordered endpoint -> model -> fs to respect the dependency chain.
    # (data_sources intentionally omitted -- see DataSource note above.)
    tuples = []
    for key in ("endpoints", "models", "feature_sets"):
        for name in result[key]:
            tuples.append((BULK_DELETE_TYPE[key], name))

    print("# --- copy/paste into bulk_delete() ---")
    print("from workbench.utils.bulk_utils import bulk_delete")
    if tuples:
        print("orphans_to_delete = [")
        for item_type, name in tuples:
            print(f'    ("{item_type}", "{name}"),')
        print("]")
    else:
        print("orphans_to_delete = []")
    print("bulk_delete(orphans_to_delete)")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find orphan artifacts not in any active pipeline")
    parser.add_argument(
        "--pipelines-path",
        default=".",
        help="Local dir (or s3:// prefix) with pipelines.json files (default: current dir)",
    )
    parser.add_argument(
        "--full-traverse",
        action="store_true",
        help="Resolve each endpoint's model via get_input() (accurate but slow); "
        "default assumes endpoint name == model name",
    )
    args = parser.parse_args()

    find_orphans(pipelines_path=args.pipelines_path, full_traverse=args.full_traverse)
