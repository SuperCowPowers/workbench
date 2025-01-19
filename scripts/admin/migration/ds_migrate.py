from workbench.api import Meta, DataSource

meta = Meta()

# Loop through the data sources and onboard
for ds_name in meta.data_sources()["Name"].values:
    ds = DataSource(ds_name)
    ds.onboard()

    # Flip over specific metadata
    params = ds.aws_meta()["Parameters"]
    ds.set_input(params.get("sageworks_input", "unknown"))
    ds.set_tags(params.get("sageworks_tags", "unknown"))
    ds.set_owner(params.get("sageworks_owner", "unknown"))
