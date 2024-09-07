import awswrangler as wr
from sageworks.aws_service_broker.aws_service_connectors.data_catalog import DataCatalog


def delete_invalid_views(database_scope):
    """Delete invalid views from the AWS Data Catalog"""

    # Create an instance of the DataCatalog
    data_catalog = DataCatalog(database_scope)
    data_catalog.refresh()

    # Check if we can connect to the AWS Data Catalog
    if not data_catalog.check():
        print("Error connecting to AWS Data Catalog")
        return

    # Grab the boto3 session from the DataCatalog
    boto3_session = data_catalog.boto3_session

    # Okay now delete invalid views for each database
    for database in data_catalog.database_scope:
        # Get all the views in this database
        views = data_catalog.get_views(database)

        # Query the view, if we get an 'INVALID_VIEW' error, delete the view
        for view in views:
            try:
                # Attempt a simple query to check if the view is valid
                query = f"SELECT * FROM {view} limit 0"
                wr.athena.read_sql_query(sql=query, database=database, ctas_approach=False, boto3_session=boto3_session)
            except Exception as e:
                if "INVALID_VIEW" in str(e):
                    print(f"Deleting invalid view: {view}...")
                    wr.catalog.delete_table_if_exists(database, view, boto3_session=boto3_session)
                else:
                    print(f"Error querying view {view}: {e}")


if __name__ == "__main__":
    database_scope = ["sageworks", "sagemaker_featurestore"]
    delete_invalid_views(database_scope)
