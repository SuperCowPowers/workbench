from workbench.api import Meta, DataSource

meta = Meta()


# Loop through the data sources and onboard
for ds_name in meta.data_sources()["Name"].values:

    ds = DataSource(ds_name)

    # Onboard the data source
    ds.onboard()
