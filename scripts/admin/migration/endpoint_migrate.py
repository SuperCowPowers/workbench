from workbench.api import Meta, Endpoint

meta = Meta()

# Loop through the Models and update the metadata
for name in meta.endpoints()["Name"].values:
    end = Endpoint(name)

    # Loop over the metadata and migrate
    meta = end.workbench_meta()

    # Build new metadata
    new_meta = {}
    for key, value in meta.items():
        if "sageworks" in key:
            print(f"Replacing {key} with {key.replace('sageworks', 'workbench')}")
            new_meta[key.replace("sageworks", "workbench")] = value

    # Add new meta
    end.upsert_workbench_meta(new_meta)

    # Now Delete all the old keys
    for key in meta.keys():
        if "sageworks" in key:
            end.delete_metadata(key)
