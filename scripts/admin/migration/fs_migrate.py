from workbench.api import Meta, FeatureSet

meta = Meta()

# Loop through the FeatureSets and onboard
for fs_name in meta.feature_sets()["Feature Group"].values:
    fs = FeatureSet(fs_name)
    fs.onboard()

    # Flip over specific metadata
    meta = fs.workbench_meta()
    fs.set_input(meta.get("sageworks_input", "unknown"))
    fs.set_tags(meta.get("sageworks_tags", "unknown").split("::"))
    fs.set_owner(meta.get("sageworks_owner", "unknown"))

    # Delete specific metadata
    keys = ["sageworks_input", "sageworks_tags", "sageworks_owner", "sageworks_status", "sageworks_health_tags"]
    for key in keys:
        fs.delete_metadata(key)
