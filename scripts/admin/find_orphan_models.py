"""Find orphan models (models whose endpoints have been 'stolen' by another model)"""

import argparse
from workbench.api import Meta, Model, Endpoint


def find_orphan_models(delete: bool = False):
    """Find models that think they have endpoints but actually don't.

    This can happen when a new model 'steals' an endpoint:
        modelA -> EndpointA
        modelB -> EndpointA  (modelB steals EndpointA from modelA)

    So modelA thinks it has an endpoint but the endpoint now points to modelB.

    Args:
        delete: If True, delete the orphan models. Default is False (just list them).
    """
    meta = Meta()

    # Get all models and endpoints
    models_df = meta.models()
    endpoints_df = meta.endpoints()

    # Build a map of endpoint -> actual model (from the endpoint's perspective)
    endpoint_to_model = {}
    for endpoint_name in endpoints_df["Name"].values:
        endpoint = Endpoint(endpoint_name)
        actual_model = endpoint.get_input()
        endpoint_to_model[endpoint_name] = actual_model

    # Find orphan models (stolen endpoints or no endpoints at all)
    orphans = []
    for model_name in models_df["Model Group"].values:
        model = Model(model_name)
        registered_endpoints = model.endpoints()

        # Models with no endpoints are orphans
        if not registered_endpoints:
            orphans.append(
                {
                    "model": model_name,
                    "reason": "no_endpoints",
                }
            )
            continue

        # Check each endpoint the model thinks it has
        for endpoint_name in registered_endpoints:
            actual_model = endpoint_to_model.get(endpoint_name)
            if actual_model and actual_model != model_name:
                orphans.append(
                    {
                        "model": model_name,
                        "reason": "stolen",
                        "claimed_endpoint": endpoint_name,
                        "actual_owner": actual_model,
                    }
                )

    # Report findings
    if not orphans:
        print("No orphan models found.")
        return

    print(f"Found {len(orphans)} orphan model(s):\n")
    for orphan in orphans:
        print(f"  Model: {orphan['model']}")
        if orphan["reason"] == "no_endpoints":
            print("    Reason: No endpoints registered")
        else:
            print("    Reason: Endpoint stolen")
            print(f"    Claims endpoint: {orphan['claimed_endpoint']}")
            print(f"    Actual owner: {orphan['actual_owner']}")
        print()

    # Delete if requested
    if delete:
        print("Deleting orphan models...")
        deleted_models = set()
        for orphan in orphans:
            model_name = orphan["model"]
            if model_name not in deleted_models:
                print(f"  Deleting {model_name}...")
                model = Model(model_name)
                model.delete()
                deleted_models.add(model_name)
        print(f"Deleted {len(deleted_models)} model(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find orphan models (models without real endpoints)")
    parser.add_argument("--delete", action="store_true", help="Delete orphan models (default: just list them)")
    args = parser.parse_args()

    find_orphan_models(delete=args.delete)
