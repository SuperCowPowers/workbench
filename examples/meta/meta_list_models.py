from pprint import pprint
from workbench.api import Meta

# Create our Meta Class and get a list of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get more details data on the Models
model_names = model_df["Model Group"].tolist()
for name in model_names:
    pprint(meta.model(name))
