import awswrangler as wr
from workbench.api import Meta


def delete_invalid_views(database_scope):
    """Delete invalid views from the AWS Data Catalog"""

    # Create an instance of the Meta class
    meta = Meta()

    # Grab the boto3 session from the DataCatalog
    boto3_session = meta.boto3_session

    # Okay now delete invalid views for each database
    for database in database_scope:
        # Get all the views in this database
        views = meta.views(database)

        # Query the view, if we get an 'INVALID_VIEW' error, delete the view
        for view in views["Name"]:
            try:
                # Attempt a simple query to check if the view is valid
                print(f"Checking view: {view}...")
                query = f'SELECT * FROM "{view}" limit 0'
                wr.athena.read_sql_query(sql=query, database=database, ctas_approach=False, boto3_session=boto3_session)
            except Exception as e:
                if "INVALID_VIEW" in str(e):
                    print(f"\tDELETING invalid view: {view}...")
                    wr.catalog.delete_table_if_exists(database, view, boto3_session=boto3_session)
                else:
                    print(f"Error querying view {view}: {e}")


if __name__ == "__main__":
    database_scope = ["workbench", "sagemaker_featurestore"]
    delete_invalid_views(database_scope)
