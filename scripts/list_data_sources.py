"""Show a list of the SageWorks data sources in the AWS catalog database"""
import sys
import argparse

# SageWorks Imports
from sageworks.utils.glue_meta import GlueMeta


# Class: List Tags
class ListTags:
    """ListTags:"""

    def __init__(self, catalog_database: str):
        # Our Glue Meta Class takes care of the low level details
        self.glue_meta = GlueMeta(catalog_database, skip_ignore=False)

    def list_tags(self):
        """List all the files/tags for the tables in this AWS catalog database"""
        catalog_table_data = self.glue_meta.get_table_data()
        for name, table_info in catalog_table_data.items():
            print(f"\nData Source: {name}")
            for key, value in table_info.items():
                print(f"\t{key}: {value}")


if __name__ == "__main__":
    # Collect args from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=str,
        default="sageworks",
        help="The AWS catalog database name",
    )
    args, commands = parser.parse_known_args()

    # Check for unknown args
    if commands:
        print("Unrecognized args: %s" % commands)
        sys.exit(1)

    # Create our Class and List the Tags
    list_tags = ListTags(args.database)
    list_tags.list_tags()
