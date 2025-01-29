"""DNS Data To Features: Custom Script
This is a custom script that knows about DNS data specifics"""

# Workbench Imports
from workbench.core.transforms.data_to_features.heavy.chunk.data_to_features_chunk import (
    DataToFeaturesChunk,
)


# Create the DataToFeaturesChunk Class
input_uuid = "heavy_dns"
output_uuid = "dns_features_0"
data_to_features = DataToFeaturesChunk(input_uuid, output_uuid)

# Set the output tags
data_to_features.set_output_tags(["dns", "heavy"])

# Construct the query of the fields that we want in our feature set
id_field = "flow_id"
event_time_field = "timestamp"
fields = [
    id_field,
    event_time_field,
    "in_iface",
    "proto",
    "dns_type",
    "dns_rrtype",
    "dns_rrname",
    "dns_flags",
    "dns_rcode",
]
query = f"SELECT {', '.join(fields)} FROM heavy_dns limit 100000"
print(query)

# Now actually perform the DataSource to FeatureSet transform
data_to_features.transform(query=query, id_column=id_field, event_time_column=event_time_field)
