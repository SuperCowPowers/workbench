from workbench.api import Meta, Model

# Create our Meta Class and get a list of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get the performance metrics for each Model
model_names = model_df["Model Group"].tolist()
for name in model_names[:5]:
    print(f"\n\nModel: {name}")
    performance_metrics = Model(name).get_inference_metrics()
    print(f"\tPerformance Metrics: {performance_metrics}")
