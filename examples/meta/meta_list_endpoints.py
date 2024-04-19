from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Endpoints
meta = Meta()
endpoints = meta.endpoints()
print(f"Number of Endpoints: {len(endpoints)}")
print(endpoints)

# Get more details data on the Endpoints
endpoints_deep = meta.endpoints_deep()
for name, info in endpoints_deep.items():
    print(name)
    print(info.keys())
