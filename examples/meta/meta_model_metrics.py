from workbench.api import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get more details data on the Models
model_names = model_df["Model Group"].tolist()
for name in model_names[:5]:
    model_details = meta.model(name)
    print(f"\n\nModel: {name}")
    performance_metrics = model_details["workbench_meta"]["workbench_inference_metrics"]
    print(f"\tPerformance Metrics: {performance_metrics}")
