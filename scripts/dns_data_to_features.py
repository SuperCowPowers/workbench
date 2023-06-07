"""DNS Data To Features: Custom Script
   This is a custom script that knows about DNS data specifics"""

# SageWorks Imports
from sageworks.transforms.data_to_features.heavy.chunk.data_to_features_chunk import DataToFeaturesChunk


# Create the DataToFeaturesChunk Class
input_uuid = "heavy_dns"
output_uuid = "dns_features_2"
data_to_features = DataToFeaturesChunk(input_uuid, output_uuid)

# Set the output tags
data_to_features.set_output_tags(["dns", "heavy"])

# Construct the query of the fields that we want in our feature set
fields = ["timestamp", "flow_id_long", "in_iface", "proto", "dns_type", "dns_rrtype", "dns_flags", "dns_rcode"]
query = f"SELECT {', '.join(fields)} FROM heavy_dns limit 100000"

# Now actually perform the DataSource to FeatureSet transform
data_to_features.transform(query=query, id_column="flow_id_long", event_time_column="timestamp")
