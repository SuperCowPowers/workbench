import time
from workbench.api import Meta, Model, ModelType

meta = Meta()


# Check if artifact has sageworks metadata
def has_sageworks_meta(meta: dict) -> bool:
    return any("sageworks" in key for key in meta)


# Loop through the Models and update the metadata
for model_name in meta.models()["Model Group"].values:
    m = Model(model_name)

    # Loop over the metadata and migrate
    meta = m.workbench_meta()

    # Check if artifact has sageworks metadata
    if not has_sageworks_meta(meta):
        print(f"Skipping {model_name} as it does not have sageworks metadata")
        continue

    # Build new metadata
    print(f"Starting migration for {model_name}...")
    new_meta = {}
    for key, value in meta.items():
        if "sageworks" in key:
            print(f"Replacing {key} with {key.replace('sageworks', 'workbench')}")
            new_meta[key.replace("sageworks", "workbench")] = value

    # Now Delete all the old keys
    for key in meta.keys():
        if "sageworks" in key:
            m.delete_metadata(key)

    # Add new meta
    m.upsert_workbench_meta(new_meta)
    print(f"Migrating done for {model_name}...")
    time.sleep(5)
