from sageworks.api.meta import Meta

# Create our Meta Class and get a summary of our Models
meta = Meta()
models = meta.models()

# Print out the summary of our Models
for name, info in models.items():
    print(f"{name}")
    latest = info[0]  # We get a list of models, so we only want the latest
    print(f"\tARN: {latest['ModelPackageGroupArn']}")
    print(f"\tDescription: {latest['ModelPackageDescription']}")
    print(f"\tTags: {latest['sageworks_meta']['sageworks_tags']}")
    performance_metrics = latest["sageworks_meta"]["sageworks_inference_metrics"]
    print(f"\tPerformance Metrics:")
    for metric in performance_metrics.keys():
        print(f"\t\t{metric}: {performance_metrics[metric]}")
