from typing import Dict
from workbench.api import Meta, DataSource

meta = Meta()


def replace_key_substrings(input_dict: Dict[str, str]) -> Dict[str, str]:
    old_substring = "sageworks"
    new_substring = "workbench"
    return {key.replace(old_substring, new_substring): value for key, value in input_dict.items()}


# Loop through the data sources and migrate the metadata
for ds_name in meta.data_sources()["Name"].values:
    ds = DataSource(ds_name)

    # Migrate the metadata
    print(f"*** Migrating Metadata for {ds_name} ***")
    ds_meta = ds.aws_meta()["Parameters"]
    new_meta = replace_key_substrings(ds_meta)
    print(new_meta.keys())

    # Update the metadata
    ds.upsert_workbench_meta(new_meta)
