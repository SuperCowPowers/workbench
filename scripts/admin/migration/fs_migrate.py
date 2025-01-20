from workbench.api import Meta, FeatureSet

meta = Meta()


# Check if artifact has sageworks metadata
def has_sageworks_meta(meta: dict) -> bool:
    return any("sageworks" in key for key in meta)


# Loop through the FeatureSets and onboard
for fs_name in meta.feature_sets()["Feature Group"].values:
    fs = FeatureSet(fs_name)

    # Check if artifact has sageworks metadata
    if not has_sageworks_meta(fs.workbench_meta()):
        print(f"Skipping {fs_name} as it does not have sageworks metadata")
        continue

    # Onboard the feature set
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
