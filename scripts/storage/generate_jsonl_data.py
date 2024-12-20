import json
import boto3
import random
import string
import gzip
import os
from datetime import datetime, timezone

# Workbench Imports
from workbench.utils.datetime_utils import datetime_to_iso8601

# Set up S3 client
s3 = boto3.client("s3")

# Set up some variables
num_files = 10  # Number of files to create
num_records_per_file = 1000  # Number of records to create in each file


# Define a function to generate random data
def generate_data():
    return {
        "id": "".join(random.choices(string.ascii_letters + string.digits, k=8)),
        "name": "".join(random.choices(string.ascii_letters, k=10)),
        "age": random.randint(18, 65),
        "address": "".join(random.choices(string.ascii_letters + string.digits + ", .", k=20)),
        "date": datetime_to_iso8601(datetime.now(timezone.utc)),
    }


# Loop through and generate the files
for i in range(num_files):
    # Set up the file name
    filename = f"data_{i}.jsonl"
    print(filename)

    # Generate the records
    records = []
    for j in range(num_records_per_file):
        records.append(generate_data())

    # Write the records to the file in JSONL format
    with open(filename, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Compress the file and save as gzipped file
    with open(filename, "rb") as f_in:
        with gzip.open(f"{filename}.gz", "wb") as f_out:
            f_out.writelines(f_in)

    # Upload the file to S3
    s3_file_path = f"incoming-data/jsonl-data/{filename}.gz"
    s3.upload_file(f"{filename}.gz", "scp-workbench-artifacts", s3_file_path)

    # Delete the local copy of the file
    os.remove(filename)
    os.remove(f"{filename}.gz")

print("Done!")
