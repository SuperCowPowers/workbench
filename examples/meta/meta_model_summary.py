from workbench.api import Meta

# Create our Meta Class and get a summary of our Models
meta = Meta()
model_df = meta.models()

print(f"Number of Models: {len(model_df)}")
print(model_df)

# Get a detailed summary (Type, Framework, Health, Owner, Status, ...)
model_details_df = meta.models(details=True)
print("\n\nModel Details Summary:")
print(model_details_df)
