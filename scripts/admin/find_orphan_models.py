"""Find orphan models and endpoints"""

import argparse
from workbench.api import Meta, Model, Endpoint


def find_orphans(delete: bool = False):
    """Find orphan models and endpoints.

    Orphan models:
        - Models with no endpoints registered
        - Models whose endpoints have been 'stolen' by another model:
            modelA -> EndpointA
            modelB -> EndpointA  (modelB steals EndpointA from modelA)

    Orphan endpoints:
        - Endpoints that point to models that no longer exist

    Args:
        delete: If True, delete the orphans. Default is False (just list them).
    """
    meta = Meta()

    # Get all models and endpoints
    models_df = meta.models()
    endpoints_df = meta.endpoints()

    # Build set of existing model names
    model_names = set(models_df["Model Group"].values)

    # Build a map of endpoint -> actual model (from the endpoint's perspective)
    # Also find orphan endpoints (pointing to non-existent models)
    endpoint_to_model = {}
    orphan_endpoints = []
    for endpoint_name in endpoints_df["Name"].values:
        endpoint = Endpoint(endpoint_name)
        actual_model = endpoint.get_input()
        endpoint_to_model[endpoint_name] = actual_model

        # Check if the endpoint points to a non-existent model
        if actual_model and actual_model not in model_names:
            orphan_endpoints.append(
                {
                    "endpoint": endpoint_name,
                    "claimed_model": actual_model,
                }
            )

    # Find orphan models (stolen endpoints or no endpoints at all)
    orphan_models = []
    for model_name in models_df["Model Group"].values:
        model = Model(model_name)
        registered_endpoints = model.endpoints()

        # Models with no endpoints are orphans
        if not registered_endpoints:
            orphan_models.append(
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
                orphan_models.append(
                    {
                        "model": model_name,
                        "reason": "stolen",
                        "claimed_endpoint": endpoint_name,
                        "actual_owner": actual_model,
                    }
                )

    # Report orphan models
    if orphan_models:
        print(f"Found {len(orphan_models)} orphan model(s):\n")
        for orphan in orphan_models:
            print(f"  Model: {orphan['model']}")
            if orphan["reason"] == "no_endpoints":
                print("    Reason: No endpoints registered")
            else:
                print("    Reason: Endpoint stolen")
                print(f"    Claims endpoint: {orphan['claimed_endpoint']}")
                print(f"    Actual owner: {orphan['actual_owner']}")
            print()
    else:
        print("No orphan models found.\n")

    # Report orphan endpoints
    if orphan_endpoints:
        print(f"Found {len(orphan_endpoints)} orphan endpoint(s):\n")
        for orphan in orphan_endpoints:
            print(f"  Endpoint: {orphan['endpoint']}")
            print(f"    Points to non-existent model: {orphan['claimed_model']}")
            print()
    else:
        print("No orphan endpoints found.\n")

    # Delete if requested
    if delete:
        if orphan_models:
            print("Deleting orphan models...")
            deleted_models = set()
            for orphan in orphan_models:
                model_name = orphan["model"]
                if model_name not in deleted_models:
                    print(f"  Deleting model {model_name}...")
                    model = Model(model_name)
                    model.delete()
                    deleted_models.add(model_name)
            print(f"Deleted {len(deleted_models)} model(s).\n")

        if orphan_endpoints:
            print("Deleting orphan endpoints...")
            for orphan in orphan_endpoints:
                endpoint_name = orphan["endpoint"]
                print(f"  Deleting endpoint {endpoint_name}...")
                endpoint = Endpoint(endpoint_name)
                endpoint.delete()
            print(f"Deleted {len(orphan_endpoints)} endpoint(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find orphan models and endpoints")
    parser.add_argument("--delete", action="store_true", help="Delete orphans (default: just list them)")
    args = parser.parse_args()

    find_orphans(delete=args.delete)
