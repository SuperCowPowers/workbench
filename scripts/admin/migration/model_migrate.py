from workbench.api import Meta, Model, ModelType

meta = Meta()

# Loop through the Models and update the metadata
for model_name in meta.models()["Model Group"].values:
    m = Model(model_name)

    # Loop over the metadata and migrate
    meta = m.workbench_meta()

    # Build new metadata
    new_meta = {}
    for key, value in meta.items():
        if "sageworks" in key:
            print(f"Replacing {key} with {key.replace('sageworks', 'workbench')}")
            new_meta[key.replace("sageworks", "workbench")] = value

    # Add new meta
    m.upsert_workbench_meta(new_meta)

    # Now Delete all the old keys
    for key in meta.keys():
        if "sageworks" in key:
            m.delete_metadata(key)
