import time
from workbench.api import Meta, Endpoint

meta = Meta()


# Check if artifact has sageworks metadata
def has_sageworks_meta(meta: dict) -> bool:
    return any("sageworks" in key for key in meta)


# Loop through the Models and update the metadata
for name in meta.endpoints()["Name"].values:
    end = Endpoint(name)

    # Loop over the metadata and migrate
    meta = end.workbench_meta()

    # Check if artifact has sageworks metadata
    if not has_sageworks_meta(meta):
        print(f"Skipping {name} as it does not have sageworks metadata")
        continue

    # Build new metadata
    print(f"Starting migration for {name}...")
    new_meta = {}
    for key, value in meta.items():
        if "sageworks" in key:
            print(f"Replacing {key} with {key.replace('sageworks', 'workbench')}")
            new_meta[key.replace("sageworks", "workbench")] = value

    # Now Delete all the old keys
    for key in meta.keys():
        if "sageworks" in key:
            end.delete_metadata(key)

    # Add new meta
    end.upsert_workbench_meta(new_meta)
    print(f"Migrating done for {name}...")
    time.sleep(5)
