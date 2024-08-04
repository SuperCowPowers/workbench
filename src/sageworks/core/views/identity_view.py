"""Create Identity View: A View that shows all columns from the data source"""

import logging
from typing import Union

# SageWorks Imports
from sageworks.api import DataSource

log = logging.getLogger("sageworks")


def create_identity_view(data_source: DataSource, view_type: "ViewType"):
    """Create Identity View: A View that shows all columns from the data source

    Args:
        data_source (DataSource): The DataSource object
        view_type (ViewType): The type of view to create
    """

    # Create the view table name
    base_table = data_source.get_table_name()
    view_name = f"{base_table}_{view_type.value}"

    log.important(f"Creating Identity View {view_name}...")

    # Construct the CREATE VIEW query
    create_view_query = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT * FROM {base_table}
    """

    # Execute the CREATE VIEW query
    data_source.execute_statement(create_view_query)


if __name__ == "__main__":
    """Exercise the Training View functionality"""
    from sageworks.api import FeatureSet
    from sageworks.core.views.view import ViewType

    # Get the FeatureSet
    fs = FeatureSet("test_features")

    # Create a default display view
    create_identity_view(fs.data_source, ViewType.COMPUTATION)

