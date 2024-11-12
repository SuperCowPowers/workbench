from pprint import pprint
from sageworks.api import Meta

# Create our Meta Class and get a list of our Endpoints
meta = Meta()
endpoint_df = meta.endpoints()
print(f"Number of Endpoints: {len(endpoint_df)}")
print(endpoint_df)

# Get more details data on the Endpoints
endpoint_names = endpoint_df["Name"].tolist()
for name in endpoint_names:
    pprint(meta.endpoint(name))
