from workbench.api import Meta

# List all the models in AWS
meta = Meta()
models = meta.models()
print(f"Found {len(models)} models in AWS")
print(models)
