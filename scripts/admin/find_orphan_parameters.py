"""Find orphan parameters -- Parameter Store entries whose model group is gone.

Model deletion cleans up ``/workbench/models/<model_group>/*``, so anything left
under a model group that SageMaker no longer lists is an orphan.

Live model groups come from Meta() rather than CachedMeta(), since stale cached
metadata would report deleted groups as live and hide the orphans.

Usage:
    python find_orphan_parameters.py             List orphaned parameters (no changes)
    python find_orphan_parameters.py --delete    List them, then delete after confirmation
"""

import argparse
from collections import defaultdict

from workbench.api import Meta, ParameterStore

MODEL_PREFIX = "/workbench/models"


def group_name(param: str) -> str:
    """Extract the model group from a parameter name, or "" if it has no group segment."""
    parts = param.split("/")  # ["", "workbench", "models", <group>, ...]
    return parts[3] if len(parts) > 3 else ""


def find_orphans(param_store: ParameterStore, live_groups: set) -> dict:
    """Map each orphaned model group to the parameters stored under it."""
    orphans = defaultdict(list)
    for param in param_store.list(MODEL_PREFIX):
        group = group_name(param)
        if group and group not in live_groups:
            orphans[group].append(param)
    return dict(orphans)


def sweep(delete: bool):
    live_groups = set(Meta().models()["Model Group"])
    if not live_groups:
        # Every parameter would look orphaned, so a transient listing failure would
        # wipe the store. Refuse rather than trust an empty result.
        print("No live model groups returned — aborting rather than treating everything as orphaned.")
        return

    param_store = ParameterStore()
    orphans = find_orphans(param_store, live_groups)
    if not orphans:
        print(f"No orphaned parameters under {MODEL_PREFIX} ({len(live_groups)} live model groups).")
        return

    total = sum(len(params) for params in orphans.values())
    print(f"Found {total} orphaned parameters across {len(orphans)} deleted model groups:")
    for group, params in sorted(orphans.items()):
        print(f"  {group}")
        for param in sorted(params):
            print(f"    {param}")

    if not delete:
        print("\nRun with --delete to remove them.")
        return

    answer = input(f"\nDelete all {total} parameters? [y/N]: ").strip().lower()
    if answer not in ("y", "yes"):
        print("Aborted.")
        return

    for group, params in sorted(orphans.items()):
        for param in sorted(params):
            param_store.delete(param)
            print(f"  DELETED {param}")


def main():
    parser = argparse.ArgumentParser(description="Find and delete orphaned model parameters in the Parameter Store")
    parser.add_argument("--delete", action="store_true", help="Delete the orphans (prompts for confirmation)")
    args = parser.parse_args()
    sweep(args.delete)


if __name__ == "__main__":
    main()
