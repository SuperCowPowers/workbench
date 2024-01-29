from sageworks.api.meta import Meta

# Create our Meta Class and get a list of our Endpoints
meta = Meta()
endpoints = meta.endpoints()

# Print out the list of our Endpoints
endpoint_list = list(endpoints.keys())
print(f"Number of Endpoints: {len(endpoint_list)}")
for name, info in endpoints.items():
    print(f"{name}")
    print(f"\tStatus: {info['EndpointStatus']}")
    print(f"\tInstance: {info['InstanceType']}")
