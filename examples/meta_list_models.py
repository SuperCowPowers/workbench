from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
models = meta.models()

# Print out the list of our Models
model_list = list(models.keys())
print(f"Number of Models: {len(model_list)}")
for model_name in models.keys():
    print(f"\t{model_name}")
