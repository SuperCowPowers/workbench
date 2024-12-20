import awswrangler as wr
from workbench.api.data_source import DataSource
from pprint import pprint


def column_changes_test():
    # Create a new Data Source from an S3 Path
    source_path = "s3://workbench-public-data/common/abalone.csv"
    df = wr.s3.read_csv(source_path)
    df.rename(columns={"Diameter": "old"}, inplace=True)
    my_data = DataSource(df, "test_columns")
    pprint(my_data.summary())

    # Change the columns of the Data Source and create DataSource with same name
    df.rename(columns={"old": "new"}, inplace=True)
    my_data = DataSource(df, "test_columns")
    pprint(my_data.summary())


if __name__ == "__main__":
    column_changes_test()
