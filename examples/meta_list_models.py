from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
models = meta.models()

print(f"Number of Models: {len(models)}")
print(models)

# Get more details data on the Endpoints
models_groups = meta.models_deep()
for name, model_versions in models_groups.items():
    print(name)
