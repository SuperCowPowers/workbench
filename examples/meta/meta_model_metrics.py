from sageworks.api.meta import Meta

# Create our Meta Class to get metadata about our Models
meta = Meta()
model_info = meta.models_deep()

# Print out the summary of our Models
for name, info in model_info.items():
    print(f"{name}")
    latest = info[0]  # We get a list of models, so we only want the latest
    print(f"\tARN: {latest['ModelPackageGroupArn']}")
    print(f"\tDescription: {latest['ModelPackageDescription']}")
    print(f"\tTags: {latest['sageworks_meta']['sageworks_tags']}")
    performance_metrics = latest["sageworks_meta"]["sageworks_inference_metrics"]
    print(f"\tPerformance Metrics:")
    print(f"\t\t{performance_metrics}")
